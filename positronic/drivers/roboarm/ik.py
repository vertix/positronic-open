"""IK solvers using MuJoCo for FK/Jacobian computation.

Three solvers for reconstructing joint-space targets from recorded EE targets:
- DmControlIKSolver: wraps dm_control qpos_from_site_pose (for sim data)
- DLSIKSolver: DLS with nullspace bias and line search (ports inverse_kinematics_q0)
- DLSIKSolverWithLimits: DLS with joint limits via bounded least squares
  (ports inverse_kinematics_with_limits, using scipy instead of OSQP)
"""

import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation as ScipyRotation

from positronic.dataset import transforms

try:
    from dm_control import mujoco as dm_mujoco
    from dm_control.utils import inverse_kinematics as dm_ik
except ImportError:
    dm_mujoco = None
    dm_ik = None


def _prepare_spec(urdf_xml, control_frame):
    """Parse URDF or MJCF into an MjSpec, stripping meshes and resolving the control frame site.

    The control frame must exist in the model as a site or body. For bodies (e.g. real URDF
    with ``end_effector`` link baked in by positronic-franka), a site is added at its origin.
    """
    root = ET.fromstring(urdf_xml)
    if root.tag == 'robot':
        for link in root.findall('.//link'):
            for elem in link.findall('visual') + link.findall('collision'):
                link.remove(elem)
        urdf_xml = ET.tostring(root, encoding='unicode')
    spec = mj.MjSpec.from_string(urdf_xml)

    all_sites = {s.name for b in spec.bodies for s in b.sites}
    if control_frame in all_sites:
        return spec

    body_names = {b.name for b in spec.bodies}
    if control_frame in body_names:
        site = spec.body(control_frame).add_site()
        site.name = control_frame
        return spec

    raise ValueError(f'Control frame {control_frame!r} not found as site or body in model')


def _parse_target(target_ee_pose_vec):
    """Parse [tx,ty,tz,w,x,y,z] into position and rotation matrix."""
    R = np.zeros(9)
    mj.mju_quat2Mat(R, target_ee_pose_vec[3:7])
    return target_ee_pose_vec[:3], R.reshape(3, 3)


def _cartesian_error(pos_cur, R_cur, t_tgt, R_tgt):
    """6D Cartesian error matching cartesian_error_ from positronic-franka robot.hpp."""
    e_pos = pos_cur - t_tgt
    e_rot = -R_cur @ ScipyRotation.from_matrix(R_cur.T @ R_tgt).as_rotvec()
    return np.concatenate([e_pos, e_rot])


class _SolverBase:
    """Base for MuJoCo-based IK solvers. Handles model loading, FK, and Jacobian.

    MuJoCo objects are built lazily on first use so that the solver stays
    naturally picklable (only urdf_xml, joint_names, control_frame, and
    solver params are stored as instance state).
    """

    def __init__(self, urdf_xml, joint_names, control_frame):
        self.urdf_xml = urdf_xml
        self.joint_names = tuple(joint_names)
        self.control_frame = control_frame
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return
        spec = _prepare_spec(self.urdf_xml, self.control_frame)
        self._model = spec.compile()
        self._data = mj.MjData(self._model)
        self._site_id = mj.mj_name2id(self._model, mj.mjtObj.mjOBJ_SITE, self.control_frame)
        self._joint_qpos_ids = np.array([self._model.joint(n).qposadr.item() for n in self.joint_names])
        self._joint_dof_ids = np.array([self._model.joint(n).dofadr.item() for n in self.joint_names])
        jnt_ids = [mj.mj_name2id(self._model, mj.mjtObj.mjOBJ_JOINT, n) for n in self.joint_names]
        self._joint_lower = self._model.jnt_range[jnt_ids, 0].copy()
        self._joint_upper = self._model.jnt_range[jnt_ids, 1].copy()
        self._init_extra(spec)

    def _init_extra(self, spec):
        """Override in subclasses that need additional setup after model loading."""

    def _fk_jac(self, q):
        """Forward kinematics + Jacobian at joint positions q."""
        self._data.qpos[self._joint_qpos_ids] = q
        mj.mj_forward(self._model, self._data)
        pos = self._data.site_xpos[self._site_id].copy()
        R = self._data.site_xmat[self._site_id].reshape(3, 3).copy()
        jacp = np.zeros((3, self._model.nv))
        jacr = np.zeros((3, self._model.nv))
        mj.mj_jacSite(self._model, self._data, jacp, jacr, self._site_id)
        J = np.vstack([jacp[:, self._joint_dof_ids], jacr[:, self._joint_dof_ids]])
        return pos, R, J

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._model = None


class DmControlIKSolver(_SolverBase):
    """IK via dm_control qpos_from_site_pose. For sim data."""

    def __init__(self, urdf_xml, joint_names, control_frame):
        if dm_mujoco is None:
            raise ImportError('dm_control is required for DmControlIKSolver')
        super().__init__(urdf_xml, joint_names, control_frame)

    def _init_extra(self, spec):
        self._physics = dm_mujoco.Physics.from_xml_string(spec.to_xml())

    def solve(self, current_q, target_ee_pose_vec):
        self._ensure_model()
        self._physics.data.qpos[self._joint_qpos_ids] = current_q
        result = dm_ik.qpos_from_site_pose(
            physics=self._physics,
            site_name=self.control_frame,
            target_pos=target_ee_pose_vec[:3],
            target_quat=target_ee_pose_vec[3:7],
            joint_names=list(self.joint_names),
            rot_weight=0.5,
        )
        return result.qpos[self._joint_qpos_ids]


class DLSIKSolver(_SolverBase):
    """DLS IK with nullspace bias and backtracking line search.

    Ports inverse_kinematics_q0 from positronic-franka robot.hpp.
    """

    def __init__(
        self,
        urdf_xml,
        joint_names,
        control_frame,
        *,
        tol=1e-4,
        max_iters=150,
        min_step=1e-8,
        pinv_reg=0.03,
        nullspace_gain=0.002,
        line_search_alpha=1.0,
        line_search_beta=0.5,
        line_search_max_steps=20,
    ):
        super().__init__(urdf_xml, joint_names, control_frame)
        self.tol = tol
        self.max_iters = max_iters
        self.min_step = min_step
        self.pinv_reg = pinv_reg
        self.nullspace_gain = nullspace_gain
        self.line_search_alpha = line_search_alpha
        self.line_search_beta = line_search_beta
        self.line_search_max_steps = line_search_max_steps

    def solve(self, current_q, target_ee_pose_vec):  # noqa: C901
        self._ensure_model()
        t_tgt, R_tgt = _parse_target(target_ee_pose_vec)
        q = current_q.astype(np.float64).copy()
        n = len(q)

        for _ in range(self.max_iters):
            pos, R_cur, J = self._fk_jac(q)
            e = _cartesian_error(pos, R_cur, t_tgt, R_tgt)
            err_norm = np.linalg.norm(e)
            if err_norm < self.tol:
                break

            # DLS pseudoinverse + nullspace bias toward zero
            J_pinv = J.T @ np.linalg.inv(J @ J.T + self.pinv_reg * np.eye(6))
            dq = -J_pinv @ e + (np.eye(n) - J_pinv @ J) @ (-self.nullspace_gain * np.exp(err_norm) * q)

            # Backtracking line search
            step = self.line_search_alpha
            best_err, best_q = err_norm, q.copy()
            for _ in range(self.line_search_max_steps):
                q_trial = q + step * dq
                pos_t, R_t, _ = self._fk_jac(q_trial)
                err_trial = np.linalg.norm(_cartesian_error(pos_t, R_t, t_tgt, R_tgt))
                if err_trial < best_err:
                    best_err = err_trial
                    best_q = q_trial
                step *= self.line_search_beta
                if step < self.min_step:
                    break

            if best_err >= err_norm - 1e-9:
                break
            q = best_q

        return q


class DLSIKSolverWithLimits(_SolverBase):
    """DLS IK with joint limits via bounded least squares.

    Ports inverse_kinematics_with_limits from positronic-franka robot.hpp,
    replacing OSQP with scipy.optimize.lsq_linear.
    """

    def __init__(
        self,
        urdf_xml,
        joint_names,
        control_frame,
        *,
        tol=1e-4,
        max_iters=150,
        min_step=1e-8,
        pinv_reg=0.03,
        line_search_alpha=1.0,
    ):
        super().__init__(urdf_xml, joint_names, control_frame)
        self.tol = tol
        self.max_iters = max_iters
        self.min_step = min_step
        self.pinv_reg = pinv_reg
        self.line_search_alpha = line_search_alpha

    def solve(self, current_q, target_ee_pose_vec):
        self._ensure_model()
        t_tgt, R_tgt = _parse_target(target_ee_pose_vec)
        q = current_q.astype(np.float64).copy()
        n = len(q)
        sqrt_lam = np.sqrt(max(self.pinv_reg, 1e-6))

        for _ in range(self.max_iters):
            pos, R_cur, J = self._fk_jac(q)
            e = _cartesian_error(pos, R_cur, t_tgt, R_tgt)
            if np.linalg.norm(e) < self.tol:
                break

            # min ||[J; √λ I] dq - [-e; 0]||² s.t. joint_lower - q ≤ dq ≤ joint_upper - q
            A = np.vstack([J, sqrt_lam * np.eye(n)])
            b = np.concatenate([-e, np.zeros(n)])
            dq = lsq_linear(A, b, bounds=(self._joint_lower - q, self._joint_upper - q)).x

            dq_norm = np.linalg.norm(dq)
            if dq_norm < self.min_step:
                break

            step_scale = min(1.0, self.line_search_alpha / dq_norm) if self.line_search_alpha > 0 else 1.0
            q_next = np.clip(q + step_scale * dq, self._joint_lower, self._joint_upper)
            if np.linalg.norm(q_next - q) < self.min_step:
                break
            q = q_next

        return q


def ik_joints_from_episode(episode, solver_cls, tgt_ee_pose_key, current_q_key):
    """Episode -> Signal. Computes target joints from EE targets via IK.

    Reads 'urdf', 'joint_names', and 'control_frame' from episode statics.
    """
    solver = solver_cls(episode['urdf'], episode['joint_names'], episode['control_frame'])
    return transforms.pairwise(episode[current_q_key], episode[tgt_ee_pose_key], solver.solve)
