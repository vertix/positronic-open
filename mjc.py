from dm_control import mujoco
from dm_control.utils import inverse_kinematics as ik

import numpy as np

model = mujoco.MjModel.from_xml_path("assets/mujoco/scene.xml")
data = mujoco.MjData(model)
physics = mujoco.Physics.from_model(data)

joints = [f'joint{i}' for i in range(1, 8)]

# Define the control parameters
control_magnitude = 0.1
control_frequency = 0.5

site = model.site('end_effector')
pos = data.site_xpos[site.id]
target_pos = np.array([0.13, -0.1, 0.54])

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        result = ik.qpos_from_site_pose(
            physics=physics,
            site_name='end_effector',
            target_pos=target_pos,
            joint_names=joints,
        )
        if result.success:
            data.ctrl[:7] = result.qpos[:7]

        mujoco.mj_step(model, data)
        viewer.sync()


print('Finished')