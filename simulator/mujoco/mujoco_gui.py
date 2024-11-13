import hydra
import mujoco
import numpy as np
from omegaconf import DictConfig
import dearpygui.dearpygui as dpg

from control import MainThreadWorld, ControlSystem, control_system, utils
from geom import Transform3D
from simulator.mujoco.environment import MujocoSimulator, InverseKinematics, DesiredAction, MujocoRenderer
from tools.dataset_dumper import DatasetDumper


@control_system(
    inputs=["images"],
    input_props=["robot_position", "simulator_ts"],
    outputs=["start_episode", "end_episode", "target_grip", "target_robot_position"],
)
class DearpyguiUi(ControlSystem):
    speed = 0.002
    movement_vectors = {
        'forward': np.array([speed, 0, 0]),
        'backward': np.array([-speed, 0, 0]),
        'left': np.array([0, speed, 0]),
        'right': np.array([0, -speed, 0]),
        'up': np.array([0, 0, speed]),
        'down': np.array([0, 0, -speed]),
    }

    key_map = {
        dpg.mvKey_W: 'forward',
        dpg.mvKey_S: 'backward',
        dpg.mvKey_A: 'left',
        dpg.mvKey_D: 'right',
        dpg.mvKey_LControl: 'down',
        dpg.mvKey_LShift: 'up',
    }

    move_update_rate = 0.1

    def __init__(self, world, width, height):
        super().__init__(world)
        self.width = width
        self.height = height

        self.desired_action = None
        self.last_success_action = None
        self.actual_position = None
        self.actual_orientation = None

        self.recording = False
        self.last_move_ts = -1

        self.move_key_states = {
            'forward': False,
            'backward': False,
            'left': False,
            'right': False,
            'up': False,
            'down': False,
        }
        self.grip_state = False

        self.raw_textures = {
            'top': np.zeros((self.height, self.width, 3), dtype=np.float32),
            'side': np.zeros((self.height, self.width, 3), dtype=np.float32),
            'handcam_left': np.zeros((self.height, self.width, 3), dtype=np.float32),
            'handcam_right': np.zeros((self.height, self.width, 3), dtype=np.float32),
        }

    def update(self):
        images = self.ins.images.read_nowait()
        if images is not None:
            ts, images = images
            self.raw_textures['top'][:] = images['top'] / 255
            self.raw_textures['side'][:] = images['side'] / 255
            self.raw_textures['handcam_left'][:] = images['handcam_left'] / 255
            self.raw_textures['handcam_right'][:] = images['handcam_right'] / 255

        # set real position
        robot_position, _ts = self.ins.robot_position()
        dpg.set_value("pos", f"Position: {robot_position}")
        self.actual_position = robot_position.translation.copy()
        self.actual_orientation = robot_position.quaternion.copy()

        if self.world.now_ts - self.last_move_ts >= self.move_update_rate:
            self.move()
            self.last_move_ts = self.world.now_ts

        if self.desired_action is not None:
            target_pos = Transform3D(self.desired_action.position, self.desired_action.orientation)
            simulator_ts = self.ins.simulator_ts()
            self.outs.target_grip.write(self.desired_action.grip, simulator_ts)
            self.outs.target_robot_position.write(target_pos, simulator_ts)


    def key_down(self, sender, app_data):
        key = app_data[0]
        key = self.key_map.get(key, None)
        self.move_key_states[key] = True

    def key_release(self, sender, app_data):
        key = app_data
        key = self.key_map.get(key, None)
        self.move_key_states[key] = False

    def grab(self):
        self.grip_state = not self.grip_state

    def record_episode(self):
        if self.recording:
            self.outs.end_episode.write(True, self.world.now_ts)
        else:
            self.outs.start_episode.write(True, self.world.now_ts)

        self.recording = not self.recording

    def move(self):
        any_key_pressed = any(self.move_key_states.values())
        if not any_key_pressed:
            return

        if self.desired_action is None:
            if self.actual_position is None:
                return
            # initialize from current position
            self.desired_action = DesiredAction(
                position=self.actual_position.copy(),
                orientation=self.actual_orientation.copy(),
                grip=0.0
            )
        #print(f"Desired action: {self.desired_action}")
        self.desired_action.grip = 1.0 if self.grip_state else 0.0
        for key, vector in self.movement_vectors.items():
            # print(f"Moving {key} {self.move_key_states.get(key, False)}")

            if self.move_key_states.get(key, False):
                self.desired_action.position += vector

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
            dpg.add_text("", tag="pos")
            dpg.add_text("", tag="target")

        with dpg.handler_registry():
            dpg.add_key_down_handler(callback=self.key_down)
            dpg.add_key_release_handler(callback=self.key_release)
            dpg.add_key_press_handler(key=dpg.mvKey_G, callback=self.grab)
            dpg.add_key_press_handler(key=dpg.mvKey_R, callback=self.record_episode)

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


@hydra.main(version_base=None, config_path=".", config_name="mujoco_gui")
def main(cfg: DictConfig):
    width = cfg.mujoco.camera_width
    height = cfg.mujoco.camera_height

    model = mujoco.MjModel.from_xml_path(cfg.mujoco.model_path)
    data = mujoco.MjData(model)

    world = MainThreadWorld()

    # systems
    simulator = MujocoSimulator(
        world=world,
        model=model,
        data=data,
        simulation_rate=1 / cfg.mujoco.simulation_hz,
    )
    renderer = MujocoRenderer(world, model, data, render_resolution=(width, height), max_fps=cfg.mujoco.observation_hz)

    inverse_kinematics = InverseKinematics(world, data=data)
    window = DearpyguiUi(world, width, height)

    # wires
    simulator.ins.bind(target_grip=window.outs.target_grip,
                       actuator_values=inverse_kinematics.outs.actuator_values)

    inverse_kinematics.ins.bind(target_robot_position=window.outs.target_robot_position)

    window.ins.bind(
        images=renderer.outs.images,
        robot_position=simulator.outs.robot_position,
        simulator_ts=simulator.outs.ts,
    )

    if cfg.data_output_dir is not None:
        @utils.map_port
        def stack_images(images):
            return np.hstack([images['handcam_left'], images['handcam_right']])
        
        properties_to_dump = utils.PropDict(world, {
            'robot_joints': simulator.outs.joints,
            'robot_position.translation': simulator.outs.robot_translation,
            'robot_position.quaternion': simulator.outs.robot_quaternion,
            'ext_force_ee': simulator.outs.ext_force_ee,
            'ext_force_base': simulator.outs.ext_force_base,
            'grip': simulator.outs.grip,
            'actuator_values': simulator.outs.actuator_values,
        })

        data_dumper = DatasetDumper(world, cfg.data_output_dir)
        data_dumper.ins.bind(
            image=stack_images(renderer.outs.images),
            target_grip=window.outs.target_grip,
            target_robot_position=window.outs.target_robot_position,
            start_episode=window.outs.start_episode,
            end_episode=window.outs.end_episode,
            robot_data=properties_to_dump.prop_values,
        )


    world.run()

if __name__ == "__main__":
    main()
