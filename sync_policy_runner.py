import hydra
from omegaconf import DictConfig
import mujoco
import rerun as rr
from tqdm import tqdm

from simulator.mujoco.sim import InverseKinematics, MujocoRenderer, MujocoSimulator
from lerobot.common.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler
from inference.action import ActionDecoder
from inference.state import StateEncoder
from inference.inference import rerun_log_action, rerun_log_observation


def get_policy(checkpoint_path: str, use_temporal_ensembler: bool = False):
    policy = ACTPolicy.from_pretrained(checkpoint_path)

    if use_temporal_ensembler:
        policy.config.n_action_steps = 1
        policy.config.temporal_ensemble_coeff = 0.01
        policy.temporal_ensembler = ACTTemporalEnsembler(0.01, policy.config.chunk_size)

    return policy


@hydra.main(version_base=None, config_path="configs", config_name="sync_policy_runner")
def main(cfg: DictConfig):
    if cfg.rerun:
        rr.init("inference", spawn=False)
        if ':' in cfg.rerun:
            rr.connect(cfg.rerun)
        elif cfg.rerun is not None:
            rr.save(cfg.rerun)

    # Initialize MuJoCo environment
    model = mujoco.MjModel.from_xml_path(cfg.mujoco.model_path)
    data = mujoco.MjData(model)
    ik = InverseKinematics(data)
    renderer = MujocoRenderer(model, data, ['handcam_left', 'handcam_right'], [cfg.mujoco.camera_width, cfg.mujoco.camera_height])
    simulator = MujocoSimulator(model, data, simulation_rate=1/cfg.mujoco.simulation_hz)

    # Initialize renderer
    renderer.initialize()

    # Reset simulator to initial state
    simulator.reset()

    # Initialize policy and encoders
    policy = get_policy(cfg.inference.checkpoint_path, cfg.inference.use_temporal_ensembler)
    policy.to(cfg.inference.device)
    state_encoder = StateEncoder(hydra.utils.instantiate(cfg.inference.state))
    action_decoder = ActionDecoder(cfg.inference.action)

    steps = cfg.inference.inference_time_sec * cfg.mujoco.simulation_hz
    render_hz = cfg.mujoco.simulation_hz // cfg.mujoco.observation_hz

    for i in tqdm(range(steps)):
        simulator.step()

        # Get observations
        if i % render_hz == 0:
            rr.set_time_seconds('time', simulator.ts_sec)
            images = renderer.render()

            # Encode state
            inputs = {
                'robot_position_translation': simulator.robot_position.translation,
                'robot_position_quaternion': simulator.robot_position.quaternion,
                'robot_joints': simulator.joints,
                'ext_force_ee': simulator.ext_force_ee,
                'ext_force_base': simulator.ext_force_base,
                'grip': simulator.grip
            }
            obs = state_encoder.encode(images, inputs)
            for key in obs:
                obs[key] = obs[key].to(cfg.inference.device)

            # Get policy action
            action = policy.select_action(obs).squeeze(0).cpu().numpy()
            action_dict = action_decoder.decode(action, inputs)

            if cfg.inference.rerun:
                rerun_log_observation(simulator.ts_sec, obs)
                rerun_log_action(simulator.ts_sec, action)

            # Apply actions
            target_pos = action_dict['target_robot_position']
            joint_positions = ik.recalculate_ik(target_pos)
            if joint_positions is not None:
                simulator.set_actuator_values(joint_positions)

            if 'target_grip' in action_dict:
                simulator.set_grip(action_dict['target_grip'])


    if cfg.rerun:
        rr.disconnect()
    renderer.close()


if __name__ == "__main__":
    main()

