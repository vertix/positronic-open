import asyncio
import threading
import time

from dataclasses import dataclass
import hydra
import mujoco
import numpy as np
from omegaconf import DictConfig
import dearpygui.dearpygui as dpg

from control.utils import fps_counter
import ironic as ir
from geom import Transform3D
from ironic.utils import FPSCounter
from simulator.mujoco.environment import MujocoSimulatorCS, InverseKinematics
from simulator.mujoco.sim import MujocoRenderer, MujocoSimulator
from tools.dataset_dumper import DatasetDumper


def _set_image_uint8_to_float32(target, source):
    target[:] = source
    target[:] /= 255


@dataclass
class DesiredAction:
    position: np.ndarray
    orientation: np.ndarray
    grip: float



@ir.ironic_system(
    input_ports=["images"],
    input_props=["robot_position"],
    output_ports=["start_tracking", "stop_tracking", "gripper_target_grasp", "robot_target_position", "reset"],
)
class DearpyguiUi(ir.ControlSystem):
    speed_meters_per_second = 0.1
    movement_vectors = {
        'forward': np.array([speed_meters_per_second, 0, 0]),
        'backward': np.array([-speed_meters_per_second, 0, 0]),
        'left': np.array([0, speed_meters_per_second, 0]),
        'right': np.array([0, -speed_meters_per_second, 0]),
        'up': np.array([0, 0, speed_meters_per_second]),
        'down': np.array([0, 0, -speed_meters_per_second]),
    }

    key_map = {
        dpg.mvKey_W: 'forward',
        dpg.mvKey_S: 'backward',
        dpg.mvKey_A: 'left',
        dpg.mvKey_D: 'right',
        dpg.mvKey_LControl: 'down',
        dpg.mvKey_LShift: 'up',
    }

    def __init__(self, width, height, episode_metadata: dict = None, initial_position: Transform3D = None):
        super().__init__()
        self.width = width
        self.height = height
        self.episode_metadata = episode_metadata or {}

        self.ui_thread = threading.Thread(target=self.ui_thread_main, daemon=True)
        self.ui_stop_event = threading.Event()
        self.render_lock = threading.Lock()
        self.swap_buffer_lock = threading.Lock()
        self.loop = asyncio.get_running_loop()

        self.initial_position = DesiredAction(
            position=initial_position.translation.copy(),
            orientation=initial_position.quaternion.copy(),
            grip=0.0
        )
        self._reset_desired_action()
        self.recording = False
        self.last_move_ts = None

        self.move_key_states = {
            'forward': False,
            'backward': False,
            'left': False,
            'right': False,
            'up': False,
            'down': False,
        }

        self.raw_textures = {
            'top': np.ones((self.height, self.width, 4), dtype=np.float32),
            'side': np.ones((self.height, self.width, 4), dtype=np.float32),
            'handcam_left': np.ones((self.height, self.width, 4), dtype=np.float32),
            'handcam_right': np.ones((self.height, self.width, 4), dtype=np.float32),
        }

        self.second_buffer = {
            'top': np.zeros((self.height, self.width, 3), dtype=np.float32),
            'side': np.zeros((self.height, self.width, 3), dtype=np.float32),
            'handcam_left': np.zeros((self.height, self.width, 3), dtype=np.float32),
            'handcam_right': np.zeros((self.height, self.width, 3), dtype=np.float32),
        }

    def _reset_desired_action(self):
        self.desired_action = DesiredAction(
            position=self.initial_position.position.copy(),
            orientation=self.initial_position.orientation.copy(),
            grip=self.initial_position.grip,
        )
    @fps_counter("update", report_every_sec=10.0)
    async def update(self):
        self.move()

        target_pos = Transform3D(self.desired_action.position, self.desired_action.orientation)

        await self.outs.gripper_target_grasp.write_message(self.desired_action.grip)
        await self.outs.robot_target_position.write_message(target_pos)

        robot_position = await self.ins.robot_position()
        dpg.set_value("robot_position", f"Robot Translation: {robot_position.data.translation}\n"
                                      f"Robot Quaternion: {robot_position.data.quaternion}")

    def key_down(self, sender, app_data):
        key = app_data[0]
        key = self.key_map.get(key, None)
        if key is not None:
            self.move_key_states[key] = True

    def key_release(self, sender, app_data):
        key = app_data
        key = self.key_map.get(key, None)
        if key is not None:
            self.move_key_states[key] = False

    def grab(self):
        self.desired_action.grip = 1.0 - self.desired_action.grip

    def switch_recording(self):
        self.loop.create_task(self._switch_recording())

    async def _switch_recording(self):
        if self.recording:
            await self.outs.stop_tracking.write_message(self.episode_metadata)
        else:
            await self.outs.reset.write_message(True)
            self._reset_desired_action()
            await self.outs.start_tracking.write_message(True)

        self.recording = not self.recording

    def move(self):
        time_since_last_move = ir.system_clock() - self.last_move_ts if self.last_move_ts is not None else 0
        time_since_last_move /= 1000000000

        for key, vector in self.movement_vectors.items():
            if self.move_key_states.get(key, False):
                self.desired_action.position += vector * time_since_last_move

        self.last_move_ts = ir.system_clock()

        dpg.set_value("target",
                      f"Target Position: {self.desired_action.position}\n"
                      f"Target Quat: {self.desired_action.orientation}\n"
                      f"Target Grip: {self.desired_action.grip}")

    @ir.on_message('images')
    async def on_images(self, message: ir.Message):
        images = message.data
        with self.swap_buffer_lock:
            _set_image_uint8_to_float32(self.second_buffer['top'], images['top'])
            _set_image_uint8_to_float32(self.second_buffer['side'], images['side'])
            _set_image_uint8_to_float32(self.second_buffer['handcam_left'], images['handcam_left'])
            _set_image_uint8_to_float32(self.second_buffer['handcam_right'], images['handcam_right'])

    def ui_thread_main(self):
        dpg.create_context()
        with dpg.texture_registry():

            dpg.add_raw_texture(width=self.width, height=self.height, tag="top", format=dpg.mvFormat_Float_rgba, default_value=self.raw_textures['top'])
            dpg.add_raw_texture(width=self.width, height=self.height, tag="side", format=dpg.mvFormat_Float_rgba, default_value=self.raw_textures['side'])
            dpg.add_raw_texture(width=self.width, height=self.height, tag="handcam_left", format=dpg.mvFormat_Float_rgba, default_value=self.raw_textures['handcam_left'])
            dpg.add_raw_texture(width=self.width, height=self.height, tag="handcam_right", format=dpg.mvFormat_Float_rgba, default_value=self.raw_textures['handcam_right'])

        with dpg.window(label="Robot"):
            with dpg.table(header_row=False):
                dpg.add_table_column()
                dpg.add_table_column()
                with dpg.table_row():
                    dpg.add_image("handcam_left")
                    dpg.add_image("handcam_right")
                with dpg.table_row():
                    dpg.add_image("top")
                    dpg.add_image("side")
        with dpg.window(label="Info"):
            dpg.add_text("", tag="target")
            dpg.add_text("", tag="robot_position")

        with dpg.handler_registry():
            dpg.add_key_down_handler(callback=self.key_down)
            dpg.add_key_release_handler(callback=self.key_release)
            dpg.add_key_press_handler(key=dpg.mvKey_G, callback=self.grab)
            dpg.add_key_press_handler(key=dpg.mvKey_R, callback=self.switch_recording)

        dpg.create_viewport(
            title='Custom Title', width=800, height=600
        )
        dpg.setup_dearpygui()
        dpg.show_viewport(maximized=True)

        fps_counter = FPSCounter("UI")

        while not self.ui_stop_event.is_set() and dpg.is_dearpygui_running():
            with self.swap_buffer_lock:
                for key in self.raw_textures:
                    self.raw_textures[key][:, :, :3] = self.second_buffer[key]
            dpg.render_dearpygui_frame()
            fps_counter.tick()

    async def setup(self):
        self.ui_thread.start()

    async def cleanup(self):
        self.ui_stop_event.set()
        await asyncio.sleep(0.1)
        self.ui_thread.join()
        dpg.destroy_context()

    async def step(self):
        await self.update()


async def _main(cfg: DictConfig):
    width = cfg.mujoco.camera_width
    height = cfg.mujoco.camera_height

    model = mujoco.MjModel.from_xml_path(cfg.mujoco.model_path)
    data = mujoco.MjData(model)

    simulator = MujocoSimulator(model=model, data=data, simulation_rate=1 / cfg.mujoco.simulation_hz)
    renderer = MujocoRenderer(model=model, data=data, render_resolution=(width, height))
    inverse_kinematics = InverseKinematics(data=data)

    simulator.reset()
    initial_position = simulator.initial_position

    # systems
    simulator = MujocoSimulatorCS(
        simulator=simulator,
        simulation_rate=1 / cfg.mujoco.simulation_hz,
        render_rate=1 / cfg.mujoco.observation_hz,
        renderer=renderer,
        inverse_kinematics=inverse_kinematics,
    )

    episode_metadata = {
        'mujoco_model_path': cfg.mujoco.model_path,
        'simulation_hz': cfg.mujoco.simulation_hz,
    }
    window = DearpyguiUi(width, height, episode_metadata, initial_position)
    if cfg.data_output_dir is not None:

        def get_translation(position):
            return position.translation

        def get_quaternion(position):
            return position.quaternion

        def discard_images(images):
            # The idea is just to pass the pulse of images, not the data
            return 0

        properties_to_dump = ir.utils.properties_dict(
            robot_joints=simulator.outs.joints,
            robot_position_translation=ir.utils.map_property(get_translation, simulator.outs.robot_position),
            robot_position_quaternion=ir.utils.map_property(get_quaternion, simulator.outs.robot_position),
            ext_force_ee=simulator.outs.ext_force_ee,
            ext_force_base=simulator.outs.ext_force_base,
            grip=simulator.outs.grip,
            actuator_values=simulator.outs.actuator_values,
        )

        data_dumper = DatasetDumper(cfg.data_output_dir)
    composed = ir.compose(
        simulator.bind(
            gripper_target_grasp=window.outs.gripper_target_grasp,
            robot_target_position=window.outs.robot_target_position,
            reset=window.outs.reset,
        ),
        window.bind(
            images=simulator.outs.images,
            robot_position=simulator.outs.robot_position,
        ),
        data_dumper.bind(
            image=ir.utils.map_port(discard_images, simulator.outs.images),
            target_grip=window.outs.gripper_target_grasp,
            target_robot_position=window.outs.robot_target_position,
            start_episode=window.outs.start_tracking,
            end_episode=window.outs.stop_tracking,
            robot_data=properties_to_dump,
        ),
    )
    await ir.utils.run_gracefully(composed)


@hydra.main(version_base=None, config_path="configs", config_name="mujoco_gui")
def main(cfg: DictConfig):
    asyncio.run(_main(cfg))

if __name__ == "__main__":
    main()
