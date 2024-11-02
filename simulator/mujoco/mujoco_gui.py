import hydra
import mujoco
import numpy as np
from omegaconf import DictConfig

from control import MainThreadWorld, ControlSystem, control_system
from simulator.mujoco.environment import MujocoControlSystem, InverseKinematicsControlSystem, DesiredAction, \
    ObservationTransform
from tools.dataset_dumper import DatasetDumper


@control_system(
    inputs=["observation", "ik_result"],
    outputs=["desired_action", "start_episode", "end_episode"]
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
            self.dpg.set_value("pos", f"Position: {obs.position}\nQuatt: {obs.orientation}")
            self.actual_position = obs.position
            self.actual_orientation = obs.orientation

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
        self.move(np.array([0.01, 0, 0]), np.array([0, 0, 0, 0]))

    def move_bwd(self):
        self.move(np.array([-0.01, 0, 0]), np.array([0, 0, 0, 0]))

    def move_left(self):
        self.move(np.array([0, 0.01, 0]), np.array([0, 0, 0, 0]))

    def move_right(self):
        self.move(np.array([0, -0.01, 0]), np.array([0, 0, 0, 0]))

    def move_up(self):
        self.move(np.array([0, 0, 0.01]), np.array([0, 0, 0, 0]))

    def move_down(self):
        self.move(np.array([0, 0, -0.01]), np.array([0, 0, 0, 0]))

    def grab(self):
        self.move(np.array([0, 0, 0]), np.array([0, 0, 0, 0]), change_grip=True)

    def record_episode(self):
        if self.recording:
            self.outs.end_episode.write(True, self.world.now_ts)
        else:
            self.outs.start_episode.write(True, self.world.now_ts)

        self.recording = not self.recording


    def move(self, dx, dq, change_grip: bool=False):
        if self.desired_action is None:
            if self.actual_position is None:
                return
            # initialize from current position
            self.desired_action = DesiredAction(
                position=self.actual_position.copy(),
                orientation=self.actual_orientation.copy(),
                grip=0
            )

        self.desired_action.position += dx
        self.desired_action.orientation += dq

        self.dpg.set_value("target", f"Target Position: {self.desired_action.position}\nTarget Quat: {self.desired_action.orientation}")
        if change_grip:
            self.desired_action.grip = 1 - self.desired_action.grip

        self.outs.desired_action.write(self.desired_action, self.world.now_ts)


    def run(self):
        import dearpygui.dearpygui as dpg
        self.dpg = dpg

        self.dpg.create_context()
        with self.dpg.texture_registry():

            self.dpg.add_raw_texture(width=self.width, height=self.height, tag="top", format=self.dpg.mvFormat_Float_rgb, default_value=self.raw_textures['top'])
            self.dpg.add_raw_texture(width=self.width, height=self.height, tag="side", format=self.dpg.mvFormat_Float_rgb, default_value=self.raw_textures['side'])
            self.dpg.add_raw_texture(width=self.width, height=self.height, tag="handcam_left", format=self.dpg.mvFormat_Float_rgb, default_value=self.raw_textures['handcam_left'])
            self.dpg.add_raw_texture(width=self.width, height=self.height, tag="handcam_right", format=self.dpg.mvFormat_Float_rgb, default_value=self.raw_textures['handcam_right'])

        with self.dpg.window(label="Robot"):
            with self.dpg.table(header_row=False):
                self.dpg.add_table_column()
                self.dpg.add_table_column()
                with self.dpg.table_row():
                    self.dpg.add_image("handcam_left")
                    self.dpg.add_image("handcam_right")
                with self.dpg.table_row():
                    self.dpg.add_image("top")
                    self.dpg.add_image("side")
            self.dpg.add_text("", tag="pos")
            self.dpg.add_text("", tag="target")

        with self.dpg.handler_registry():
            self.dpg.add_key_press_handler(key=self.dpg.mvKey_W, callback=self.move_fwd)
            self.dpg.add_key_press_handler(key=self.dpg.mvKey_S, callback=self.move_bwd)
            self.dpg.add_key_press_handler(key=self.dpg.mvKey_A, callback=self.move_left)
            self.dpg.add_key_press_handler(key=self.dpg.mvKey_D, callback=self.move_right)
            self.dpg.add_key_press_handler(key=self.dpg.mvKey_LControl, callback=self.move_down)
            self.dpg.add_key_press_handler(key=self.dpg.mvKey_LShift, callback=self.move_up)
            self.dpg.add_key_press_handler(key=self.dpg.mvKey_G, callback=self.grab)
            self.dpg.add_key_press_handler(key=self.dpg.mvKey_R, callback=self.record_episode)

        self.dpg.create_viewport(title='Custom Title', width=800, height=600)
        self.dpg.setup_dearpygui()
        self.dpg.show_viewport()
        self.dpg.maximize_viewport()
        self.update()
        self.move(np.array([0, 0, 0]), np.array([0, 0, 0, 0]))
        # self.dpg.start_dearpygui()
        while self.dpg.is_dearpygui_running():
            if self.world.should_stop:
                break
            self.update()
            self.dpg.render_dearpygui_frame()

            # if state['env'] is not None:
            #     self.dpg.set_value("pos",
            #                   f"Position: {state['env'].data.site('end_effector').xpos}\nQuatt: {state['env'].data.body('hand').xquat}")
        self.dpg.destroy_context()
        self.world.stop_event.set()

@hydra.main(version_base=None, config_path=".", config_name="mujoco_gui")
def main(cfg: DictConfig):
    width = cfg.mujoco.camera_width
    height = cfg.mujoco.camera_height

    model = mujoco.MjModel.from_xml_path(cfg.mujoco.model_path)
    data = mujoco.MjData(model)

    world = MainThreadWorld()

    # systems
    simulator = MujocoControlSystem(world, model, data ,
                                         render_resolution=(width, height))
    inverse_kinematics = InverseKinematicsControlSystem(world, data=data)
    window = DearpyguiUi(world, width, height)
    observation_transform = ObservationTransform(world)
    data_dumper = DatasetDumper(world, cfg.data_output_dir)

    # wires
    simulator.ins.actuator_values = inverse_kinematics.outs.actuator_values

    inverse_kinematics.ins.desired_action = window.outs.desired_action

    window.ins.observation = simulator.outs.observation
    window.ins.ik_result = inverse_kinematics.outs.actuator_values

    observation_transform.ins.observation = simulator.outs.observation
    observation_transform.ins.desired_action = window.outs.desired_action

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