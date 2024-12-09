import franky
import hydra
from omegaconf import DictConfig

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

    waypoints = []
    last_q = robot.state.q
    for target in cfg.targets:
        if "joints" in target:
            waypoints.append(franky.JointWaypoint(target.joints))
        elif "translation" in target:
            pos = franky.Affine(translation=target.translation, quaternion=target.quaternion)
            q = robot.inverse_kinematics(pos, last_q)
            waypoints.append(franky.JointWaypoint(q))
            last_q = q

    robot.move(franky.JointWaypointMotion(waypoints))
    print(robot.current_pose.end_effector_pose)


if __name__ == "__main__":
    main()
