import hydra
import mujoco
import numpy as np
from omegaconf import DictConfig
import dearpygui.dearpygui as dpg

from control import MainThreadWorld, ControlSystem, control_system
from simulator.mujoco.environment import Mujoco, InverseKinematics, DesiredAction, extract_information_to_dump
from tools.dataset_dumper import DatasetDumper


@control_system(
    inputs=["observation", "ik_result"],
    input_props=["robot_position"],
    outputs=["desired_action", "start_episode", "end_episode", "target_grip", "target_robot_position"]
)
class DearpyguiUi(ControlSystem):
    def __init__(self, world, width, height):
        super().__init__(world)
        self.width = width
        self.height = height

        self.desired_action = None
        self.last_success_action = None
        self.actual_position = None
        self.actual_orientation = None

        self.recording = False

        self.raw_textures = {
            'top': np.zeros((self.height, self.width, 3), dtype=np.float32),
            'side': np.zeros((self.height, self.width, 3), dtype=np.float32),
            'handcam_left': np.zeros((self.height, self.width, 3), dtype=np.float32),
            'handcam_right': np.zeros((self.height, self.width, 3), dtype=np.float32),
        }

    def update(self):
        obs = self.ins.observation.read_nowait()
        if obs is not None:
            ts, obs = obs
            self.raw_textures['top'][:] = obs.top_image / 255
            self.raw_textures['side'][:] = obs.side_image / 255
            self.raw_textures['handcam_left'][:] = obs.handcam_left_image / 255
            self.raw_textures['handcam_right'][:] = obs.handcam_right_image / 255

        # set real position
        robot_position, _ts = self.ins.robot_position()
        dpg.set_value("pos", f"Position: {robot_position}")
        self.actual_position = robot_position.translation
        self.actual_orientation = robot_position.quaternion

        ik_result = self.ins.ik_result.read_nowait()
        if ik_result is not None:
            ts, ik_result = ik_result
            if ik_result.success:
                self.last_success_action = self.desired_action
            else:
                self.desired_action = DesiredAction(
                    position=self.last_success_action.position.copy(),
                    orientation=self.last_success_action.orientation.copy(),
                    grip=self.last_success_action.grip
                )


    def move_fwd(self):
        self.move(np.array([0.01, 0, 0]))

    def move_bwd(self):
        self.move(np.array([-0.01, 0, 0]))

    def move_left(self):
        self.move(np.array([0, 0.01, 0]))

    def move_right(self):
        self.move(np.array([0, -0.01, 0]))

    def move_up(self):
        self.move(np.array([0, 0, 0.01]))

    def move_down(self):
        self.move(np.array([0, 0, -0.01]))

    def grab(self):
        self.move(np.array([0, 0, 0]), change_grip=True)

    def record_episode(self):
        if self.recording:
            self.outs.end_episode.write(True, self.world.now_ts)
        else:
            self.outs.start_episode.write(True, self.world.now_ts)

        self.recording = not self.recording


    def move(self, dx, change_grip: bool=False):
        if self.desired_action is None:
            if self.actual_position is None:
                return
            # initialize from current position
            self.desired_action = DesiredAction(
                position=self.actual_position.copy(),
                orientation=self.actual_orientation.copy(),
                grip=0.0
            )

        self.desired_action.position += dx

        dpg.set_value("target", f"Target Position: {self.desired_action.position}\nTarget Quat: {self.desired_action.orientation}")
        if change_grip:
            self.desired_action.grip = 1.0 - self.desired_action.grip

        self.outs.desired_action.write(self.desired_action, self.world.now_ts)


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
            dpg.add_key_press_handler(key=dpg.mvKey_W, callback=self.move_fwd)
            dpg.add_key_press_handler(key=dpg.mvKey_S, callback=self.move_bwd)
            dpg.add_key_press_handler(key=dpg.mvKey_A, callback=self.move_left)
            dpg.add_key_press_handler(key=dpg.mvKey_D, callback=self.move_right)
            dpg.add_key_press_handler(key=dpg.mvKey_LControl, callback=self.move_down)
            dpg.add_key_press_handler(key=dpg.mvKey_LShift, callback=self.move_up)
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
    simulator = Mujoco(
        world=world,
        model=model,
        data=data,
        render_resolution=(width, height),
        simulation_rate=1 / cfg.mujoco.simulation_hz,
        observation_rate=1 / cfg.mujoco.observation_hz
    )
    inverse_kinematics = InverseKinematics(world, data=data)
    window = DearpyguiUi(world, width, height)
    observation_transform = extract_information_to_dump(world)

    # wires
    simulator.ins.actuator_values = inverse_kinematics.outs.actuator_values

    inverse_kinematics.ins.desired_action = window.outs.desired_action

    window.ins.bind(observation=simulator.outs.observation,
                    ik_result=inverse_kinematics.outs.actuator_values,
                    robot_position=simulator.outs.robot_position)

    observation_transform.ins.observation = simulator.outs.observation
    observation_transform.ins.desired_action = window.outs.desired_action

    if cfg.data_output_dir is not None:
        data_dumper = DatasetDumper(world, cfg.data_output_dir)
        data_dumper.ins.image = observation_transform.outs.image
        data_dumper.ins.robot_joints = observation_transform.outs.robot_joints
        data_dumper.ins.robot_position = observation_transform.outs.robot_position
        data_dumper.ins.ext_force_ee = observation_transform.outs.ext_force_ee
        data_dumper.ins.ext_force_base = observation_transform.outs.ext_force_base
        data_dumper.ins.grip = observation_transform.outs.grip
        data_dumper.ins.target_grip = observation_transform.outs.target_grip
        data_dumper.ins.target_robot_position = observation_transform.outs.target_robot_position
        data_dumper.ins.start_episode = window.outs.start_episode
        data_dumper.ins.end_episode = window.outs.end_episode

    world.run()

if __name__ == "__main__":
    main()
