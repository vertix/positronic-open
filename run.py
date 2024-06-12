from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.sensor import Camera
import numpy as np
import omni.isaac.core.utils.numpy.rotations as rot_utils
import rerun as rr  # Import rerun for logging images

rr.init("isaacsim_visualization")
rr.save("run.rrd")

def main():
    my_world = World(stage_units_in_meters=1.0)

    print('getting assets')
    assets_root_path = get_assets_root_path()
    asset_path = assets_root_path + "/Isaac/Robots/Franka/franka_alt_fingers.usd"
    print(f"asset_path: {asset_path}")
    print('adding reference to stage')
    add_reference_to_stage(asset_path, "/Franka")

    cube_2 = my_world.scene.add(
        DynamicCuboid(
            prim_path="/new_cube_2",
            name="cube_1",
            position=np.array([5.0, 3, 1.0]),
            scale=np.array([0.6, 0.5, 0.2]),
            size=1.0,
            color=np.array([255, 0, 0]),
        )
    )

    cube_3 = my_world.scene.add(
        DynamicCuboid(
            prim_path="/new_cube_3",
            name="cube_2",
            position=np.array([-5, 1, 3.0]),
            scale=np.array([0.1, 0.1, 0.1]),
            size=1.0,
            color=np.array([0, 0, 255]),
            linear_velocity=np.array([0, 0, 0.4]),
        )
    )

    camera = Camera(
        prim_path="/World/camera",
        position=np.array([0.0, 0, 3]),
        frequency=20,
        resolution=(256, 256),
        orientation=rot_utils.euler_angles_to_quats(np.array([0, 90, 0]), degrees=True),
    )

    my_world.scene.add_default_ground_plane()
    my_world.reset()
    franka = Articulation("/Franka")
    franka.initialize()
    camera.initialize()

    camera.add_motion_vectors_to_frame()
    # Get the degree of freedom (DOF) pointer for a specific joint
    dof_ptr = franka.get_dof_index("panda_joint2")

    simulation_time = 0.0
    time_step = my_world.get_physics_dt()  # Get the time step of the simulation

    while simulation_app.is_running() and simulation_time < 10.0:
        my_world.step(render=True)
        # Set joint positions to move the arm
        franka.set_joint_positions([-1.5 * (np.sin(simulation_time) + 1) / 2], [dof_ptr])

        rr.set_time_seconds("time", simulation_time)
        image = camera.get_rgba()
        if len(image.shape) == 3:
            image = image[:, :, :3]
            rr.log("camera_image", rr.Image(image).compress(jpeg_quality=95))

        if my_world.is_playing():
            if my_world.current_time_step_index == 0:
                my_world.reset()
        simulation_time += time_step

if __name__ == "__main__":
    main()
simulation_app.close()
rr.shutdown()  # Shutdown rerun
