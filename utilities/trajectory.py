import franky
import hydra
from omegaconf import DictConfig

from control.world import MainThreadWorld
from hardware.franka import Franka

@hydra.main(version_base=None, config_path=".", config_name="trajectory")
def main(cfg: DictConfig):
    robot = franky.Robot(cfg.ip, realtime_config=franky.RealtimeConfig.Ignore)
    robot.relative_dynamics_factor = cfg.relative_dynamics_factor
    robot.set_collision_behavior(
        [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
        [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
        [30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0],
        [30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0],
        [50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
        [50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
        [30.0, 30.0, 30.0, 30.0, 30.0, 30.0],
        [30.0, 30.0, 30.0, 30.0, 30.0, 30.0]
    )

    robot.recover_from_errors()
    robot.move(franky.JointWaypointMotion([
        franky.JointWaypoint([0.0,  -0.31, 0.0, -1.53, 0.0, 1.522,  0.785])]))

    print(robot.current_pose.end_effector_pose)

    waypoints = []
    for target in cfg.targets:
        pos = franky.Affine(translation=target.translation, quaternion=target.quaternion)
        waypoints.append(franky.CartesianWaypoint(pos, franky.ReferenceType.Absolute))

    robot.move(franky.CartesianWaypointMotion(waypoints))
    print(robot.current_pose.end_effector_pose)


if __name__ == "__main__":
    main()
