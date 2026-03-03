"""DreamZero LoRA fine-tuning: wraps upstream Hydra training pipeline."""

import os
import subprocess
from pathlib import Path

import configuronic as cfn
import pos3

from positronic import utils

DREAMZERO_ROOT = Path('/dreamzero')
DREAMZERO_VENV = Path('/.venv')
CACHE_DIR = Path('/root/.cache/dreamzero')

WAN_REPO = 'Wan-AI/Wan2.1-I2V-14B-480P'
UMT5_REPO = 'google/umt5-xxl'


def _download_base_weights():
    """Download Wan2.1-I2V-14B-480P and umt5-xxl tokenizer to persistent cache."""
    python = str(DREAMZERO_VENV / 'bin' / 'python')
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    wan_dir = CACHE_DIR / 'Wan2.1-I2V-14B-480P'
    umt5_dir = CACHE_DIR / 'umt5-xxl'

    if not wan_dir.exists():
        print(f'Downloading {WAN_REPO} to {wan_dir}...')
        subprocess.run(
            [
                python,
                '-c',
                f'from huggingface_hub import snapshot_download; '
                f'snapshot_download("{WAN_REPO}", local_dir="{wan_dir}")',
            ],
            check=True,
        )
    else:
        print(f'Base weights already cached at {wan_dir}')

    if not umt5_dir.exists():
        print(f'Downloading {UMT5_REPO} to {umt5_dir}...')
        subprocess.run(
            [
                python,
                '-c',
                f'from huggingface_hub import snapshot_download; '
                f'snapshot_download("{UMT5_REPO}", local_dir="{umt5_dir}")',
            ],
            check=True,
        )
    else:
        print(f'Tokenizer already cached at {umt5_dir}')

    return wan_dir, umt5_dir


@cfn.config(num_gpus=1, max_steps=100, learning_rate=1e-5, save_steps=1000, resume=False, extra_args=[])
def main(
    input_path: str,
    output_path: str,
    exp_name: str,
    num_gpus: int,
    max_steps: int,
    learning_rate: float,
    save_steps: int,
    resume: bool,
    extra_args: list[str],
):
    exp_name = str(exp_name)
    torchrun = str(DREAMZERO_VENV / 'bin' / 'torchrun')

    wan_dir, umt5_dir = _download_base_weights()

    with pos3.mirror():
        dataset_local_path = pos3.download(input_path)
        output_path = output_path.rstrip('/')
        output_dir = pos3.sync(output_path + '/' + exp_name, delete_remote=not resume)
        prefix = 'resume_metadata' if resume else 'run_metadata'
        utils.save_run_metadata(output_dir, patterns=['*.py', '*.toml'], prefix=prefix)

        command = [
            torchrun,
            f'--nproc_per_node={num_gpus}',
            'groot/vla/experiment/experiment.py',
            'data=dreamzero/droid_relative',
            f'droid_data_root={dataset_local_path}',
            'model=dreamzero/vla',
            'model/dreamzero/action_head=wan_flow_matching_action_tf',
            'model/dreamzero/transform=dreamzero_cotrain',
            'train_architecture=lora',
            f'report_to={"wandb" if os.environ.get("WANDB_API_KEY") else "none"}',
            'wandb_project=dreamzero',
            'num_frames=33',
            'action_horizon=24',
            'num_views=3',
            'num_frame_per_block=2',
            'num_action_per_block=24',
            'num_state_per_block=1',
            'image_resolution_width=320',
            'image_resolution_height=176',
            'frame_seqlen=880',
            'max_chunk_size=4',
            'per_device_train_batch_size=1',
            'bf16=true',
            'tf32=true',
            f'training_args.deepspeed={DREAMZERO_ROOT}/groot/vla/configs/deepspeed/zero2.json',
            f'training_args.learning_rate={learning_rate}',
            'training_args.warmup_ratio=0.05',
            f'max_steps={max_steps}',
            f'save_steps={save_steps}',
            f'output_dir={output_dir}',
            'save_lora_only=true',
            'save_total_limit=10',
            f'dit_version={wan_dir}',
            f'text_encoder_pretrained_path={wan_dir}/models_t5_umt5-xxl-enc-bf16.pth',
            f'image_encoder_pretrained_path={wan_dir}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth',
            f'vae_pretrained_path={wan_dir}/Wan2.1_VAE.pth',
            f'tokenizer_path={umt5_dir}',
        ]
        command.extend(extra_args)

        env = os.environ.copy()
        env['VIRTUAL_ENV'] = str(DREAMZERO_VENV)
        env['PATH'] = f'{DREAMZERO_VENV / "bin"}:{env.get("PATH", "")}'
        env['HYDRA_FULL_ERROR'] = '1'

        print(f'Running command: `{" ".join(command)}`')
        subprocess.run(command, check=True, cwd=str(DREAMZERO_ROOT), env=env)


if __name__ == '__main__':
    cfn.cli(main)
