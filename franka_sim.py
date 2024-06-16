import os

import numpy as np
import rerun as rr


from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import omni
from omni.isaac.core import World
from omni.isaac.sensor import Camera
from omni.isaac.franka import Franka, KinematicsSolver

rr.init("isaacsim_visualization")
rr.save("run.rrd")

def main():
    current_directory = os.getcwd()
    stage_path = os.path.join(current_directory, 'assets/franka_table.usda')
    omni.usd.get_context().open_stage(stage_path)
    omni.kit.app.get_app().update()

    world = World(stage_units_in_meters=1.0)
    world.initialize_physics()
    world.reset()

    stage = omni.usd.get_context().get_stage()
    franka = Franka(prim_path="/World/Franka", name="Franka")
    franka.initialize()

    controller = KinematicsSolver(franka)
    articulation_controller = franka.get_articulation_controller()

    target_position = np.array([0.5, 0.0, 0.7])
    target_orientation = np.array([0.0, 0.0, 0.0, 1.0])
    r = np.array([0, 0.05, 0.05])

    camera = Camera(prim_path="/World/Camera")
    camera.initialize()

    simulation_time = 0.0
    time_step = world.get_physics_dt()

    while simulation_app.is_running() and simulation_time < 60.0:
        world.step(render=True)

        actions, succ = controller.compute_inverse_kinematics(
            target_position=target_position + r*np.sin(simulation_time),
            target_orientation=target_orientation)
        if succ:
            articulation_controller.apply_action(actions)
        else:
            print("IK did not converge to a solution.  No action is being taken.")

        rr.set_time_seconds("time", simulation_time)
        image = camera.get_rgba()
        if len(image.shape) == 3:
            image = image[:, :, :3]
            rr.log("camera_image", rr.Image(image).compress(jpeg_quality=95))

        simulation_time += time_step

if __name__ == "__main__":
    main()
simulation_app.close()
rr.shutdown()
