"""LeRobot codecs (observation encoder | action decoder pairs)."""

from positronic.cfg import codecs

# LeRobot codec variants using codec.override for lazy composition
eepose_absolute = codecs.codec.override(
    observation=codecs.eepose_obs, action=codecs.absolute_pos_action(rotation_rep=None)
)

joints_absolute = codecs.codec.override(
    observation=codecs.joints_obs, action=codecs.absolute_pos_action(rotation_rep=None)
)

# Trajectory variants: use actual robot trajectory as action target instead of commanded targets
eepose_absolute_traj = codecs.codec.override(
    observation=codecs.eepose_obs,
    action=codecs.absolute_pos_action(rotation_rep=None, tgt_ee_pose_key='robot_state.ee_pose', tgt_grip_key='grip'),
)

joints_absolute_traj = codecs.codec.override(
    observation=codecs.joints_obs,
    action=codecs.absolute_pos_action(rotation_rep=None, tgt_ee_pose_key='robot_state.ee_pose', tgt_grip_key='grip'),
)
