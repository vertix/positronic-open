from dataclasses import dataclass
import hydra
import mujoco
import numpy as np
from omegaconf import DictConfig
import dearpygui.dearpygui as dpg

from control import MainThreadWorld, ControlSystem, control_system, utils
from geom import Transform3D
from simulator.mujoco.environment import MujocoSimulatorCS, InverseKinematics
from simulator.mujoco.sim import MujocoRenderer, MujocoSimulator
from tools.dataset_dumper import DatasetDumper


def _set_image_uint8_to_float32(target, source):
    target[:] = source.astype(np.float32)
    target[:] /= 255



@dataclass
class DesiredAction:
    position: np.ndarray
    orientation: np.ndarray
    grip: float



@control_system(
    inputs=["images"],
    outputs=["start_episode", "end_episode", "target_grip", "target_robot_position", "reset"],
)
class DearpyguiUi(ControlSystem):
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

    def __init__(self, world, width, height, episode_metadata: dict = None, initial_position: Transform3D = None):
        super().__init__(world)
        self.width = width
        self.height = height
        self.episode_metadata = episode_metadata or {}

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

    def update(self):
        images = self.ins.images.read_nowait()
        if images is not None:
            ts, images = images
            _set_image_uint8_to_float32(self.raw_textures['top'], images['top'])
            _set_image_uint8_to_float32(self.raw_textures['side'], images['side'])
            _set_image_uint8_to_float32(self.raw_textures['handcam_left'], images['handcam_left'])
            _set_image_uint8_to_float32(self.raw_textures['handcam_right'], images['handcam_right'])

        self.move()

        target_pos = Transform3D(self.desired_action.position, self.desired_action.orientation)

        self.outs.target_grip.write(self.desired_action.grip, self.world.now_ts)
        self.outs.target_robot_position.write(target_pos, self.world.now_ts)

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
        if self.recording:
            self.outs.end_episode.write(self.episode_metadata, self.world.now_ts)
        else:
            self.outs.reset.write(True, self.world.now_ts)
            self._reset_desired_action()
            self.outs.start_episode.write(True, self.world.now_ts)
            
        self.recording = not self.recording

    def move(self):
        time_since_last_move = self.world.now_ts - self.last_move_ts if self.last_move_ts is not None else 0
        time_since_last_move /= 1000

        for key, vector in self.movement_vectors.items():
            if self.move_key_states.get(key, False):
                self.desired_action.position += vector * time_since_last_move

        self.last_move_ts = self.world.now_ts

        dpg.set_value("target",
                      f"Target Position: {self.desired_action.position}\n"
                      f"Target Quat: {self.desired_action.orientation}\n"
                      f"Target Grip: {self.desired_action.grip}")

    def run(self):
        dpg.create_context()
        with dpg.texture_registry():

            dpg.add_raw_texture(width=self.width, height=self.height, tag="top", format=dpg.mvFormat_Float_rgb, default_value=self.raw_textures['top'])
            dpg.add_raw_texture(width=self.width, height=self.height, tag="side", format=dpg.mvFormat_Float_rgb, default_value=self.raw_textures['side'])
            dpg.add_raw_texture(width=self.width, height=self.height, tag="handcam_left", format=dpg.mvFormat_Float_rgb, default_value=self.raw_textures['handcam_left'])
            dpg.add_raw_texture(width=self.width, height=self.height, tag="handcam_right", format=dpg.mvFormat_Float_rgb, default_value=self.raw_textures['handcam_right'])

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

        with dpg.handler_registry():
            dpg.add_key_down_handler(callback=self.key_down)
            dpg.add_key_release_handler(callback=self.key_release)
            dpg.add_key_press_handler(key=dpg.mvKey_G, callback=self.grab)
            dpg.add_key_press_handler(key=dpg.mvKey_R, callback=self.switch_recording)

        dpg.create_viewport(
            title='Custom Title', width=800, height=600
        )
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.maximize_viewport()

        while dpg.is_dearpygui_running():
            if self.world.should_stop:
                break

            self.update()
            dpg.render_dearpygui_frame()

        dpg.destroy_context()
        self.world.stop_event.set()


@hydra.main(version_base=None, config_path="configs", config_name="mujoco_gui")
def main(cfg: DictConfig):
    width = cfg.mujoco.camera_width
    height = cfg.mujoco.camera_height

    model = mujoco.MjModel.from_xml_path(cfg.mujoco.model_path)
    data = mujoco.MjData(model)

    simulator = MujocoSimulator(model=model, data=data, simulation_rate=1 / cfg.mujoco.simulation_hz)
    renderer = MujocoRenderer(model=model, data=data, render_resolution=(width, height))
    inverse_kinematics = InverseKinematics(data=data)

    simulator.reset()
    initial_position = simulator.initial_position

    world = MainThreadWorld()

    # systems
    simulator = MujocoSimulatorCS(
        world=world,
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
    window = DearpyguiUi(world, width, height, episode_metadata, initial_position)
    
    # wires
    simulator.ins.bind(target_grip=window.outs.target_grip,
                       target_robot_position=window.outs.target_robot_position,
                       reset=window.outs.reset)

    window.ins.bind(
        images=simulator.outs.images,
    )

    if cfg.data_output_dir is not None:

        @utils.map_prop
        def get_translation(position):
            return position.translation

        @utils.map_prop
        def get_quaternion(position):
            return position.quaternion
        
        @utils.map_port
        def discard_images(images):
            # The idea is just to pass the pulse of images, not the data
            return 0

        properties_to_dump = utils.properties_dict(
            robot_joints=simulator.outs.joints,
            robot_position_translation=get_translation(simulator.outs.robot_position),
            robot_position_quaternion=get_quaternion(simulator.outs.robot_position),
            ext_force_ee=simulator.outs.ext_force_ee,
            ext_force_base=simulator.outs.ext_force_base,
            grip=simulator.outs.grip,
            actuator_values=simulator.outs.actuator_values,
        )

        data_dumper = DatasetDumper(world, cfg.data_output_dir)
        data_dumper.ins.bind(
            image=discard_images(simulator.outs.images),  # TODO: allow dumper to dump on any port
            target_grip=window.outs.target_grip,
            target_robot_position=window.outs.target_robot_position,
            start_episode=window.outs.start_episode,
            end_episode=window.outs.end_episode,
            robot_data=properties_to_dump,
        )


    world.run()

if __name__ == "__main__":
    main()
