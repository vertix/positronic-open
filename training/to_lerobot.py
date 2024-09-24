import argparse
import logging
from pathlib import Path

import cv2
import torch
import tqdm
import numpy as np
import hydra
from omegaconf import DictConfig

from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub.aloha_hdf5_format import to_hf_dataset
from lerobot.common.datasets.push_dataset_to_hub.utils import concatenate_episodes, get_default_encoding
from lerobot.common.datasets.utils import calculate_episode_data_index
from lerobot.common.datasets.video_utils import encode_video_frames
from lerobot.scripts.push_dataset_to_hub import save_meta_data
from lerobot.common.datasets.compute_stats import compute_stats

def convert_to_lerobot_dataset(input_dir, output_dir, fps=30, video=True, run_compute_stats=True):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "videos").mkdir(exist_ok=True)
    (output_dir / "episodes").mkdir(exist_ok=True)

    # Process each episode file
    episode_files = sorted([f for f in input_dir.glob('*.pt')])

    all_episodes_data = []

    for episode_idx, episode_file in enumerate(tqdm.tqdm(episode_files, desc="Processing episodes")):
        episode_data = torch.load(episode_file)

        ep_dict = {}

        # Process images
        for side in ['left', 'right']:
            w = episode_data['image'].shape[2]
            if side == 'left':
                images = episode_data['image'][:, :, :w //2, :].numpy()
            else:
                images = episode_data['image'][:, :, w //2:, :].numpy()

            video_filename = f"episode_{episode_idx:04d}_{side}.mp4"
            video_path = output_dir / "videos" / video_filename

            # Save frames as temporary PNG files
            temp_dir = output_dir / "temp_frames" / f"episode_{episode_idx:04d}_{side}"
            temp_dir.mkdir(parents=True, exist_ok=True)
            for i, img in enumerate(tqdm.tqdm(images, desc=f"Saving frames for {side}")):
                img_path = temp_dir / f"frame_{i:06d}.png"
                if img.shape[0] == 3:  # If image is in CHW format
                    img = np.transpose(img, (1, 2, 0))  # CHW to HWC
                elif img.shape[2] != 3:  # If image is not in RGB format
                    raise ValueError(f"Unexpected image shape: {img.shape}")
                cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            encode_video_frames(temp_dir, video_path, fps, overwrite=True)  # Encode video using ffmpeg

            for file in temp_dir.glob("*.png"):
                file.unlink()
            temp_dir.rmdir()
            ep_dict[f"observation.images.{side}"] = [{"path": f"videos/{video_filename}", "timestamp": i / fps} for i in range(len(images))]

        num_frames = len(episode_data['time'])  # TODO: Check if we really need a timestamp and not seconds

        # Process state
        # TODO: Should we normalise it?
        state = torch.zeros((num_frames, 3 + 4 + 7 + 6 + 6 + 1))  # position, rotation, joint angles, ee force, base force, gripper
        state[:, :3] = episode_data['robot_position_trans']
        state[:, 3:7] = episode_data['robot_position_quat']
        state[:, 7:14] = episode_data['robot_joints']
        state[:, 14:20] = episode_data['ee_force']
        state[:, 20:26] = episode_data['base_force']
        state[:, 26] = episode_data['grip']
        ep_dict['observation.state'] = state

        action = torch.zeros((num_frames, 3 + 4 + 1))  # position, rotation, gripper
        action[:, :3] = episode_data['target_robot_position_trans']
        action[:, 3:7] = episode_data['target_robot_position_quat']
        action[:, 7] = episode_data['target_grip']
        ep_dict['action'] = action

        ep_dict["episode_index"] = torch.tensor([episode_idx] * num_frames)
        ep_dict["frame_index"] = torch.arange(0, num_frames, 1)
        ep_dict["timestamp"] = torch.tensor(episode_data['time'])

        done = torch.zeros(num_frames, dtype=torch.bool)
        # done[-1] = True
        ep_dict["next.done"] = done

        all_episodes_data.append(ep_dict)

        # Save individual episode
        ep_path = output_dir / "episodes" / f"episode_{episode_idx}.pth"
        torch.save(ep_dict, ep_path)

    # Concatenate all episodes
    data_dict = concatenate_episodes(all_episodes_data)
    total_frames = data_dict["frame_index"].shape[0]
    data_dict["index"] = torch.arange(0, total_frames, 1)

    # Create HuggingFace dataset
    hf_dataset = to_hf_dataset(data_dict, video)
    episode_data_index = calculate_episode_data_index(hf_dataset)

    info = {
        "codebase_version": CODEBASE_VERSION,
        "fps": fps,
        "video": video,
    }
    if video:
        info["encoding"] = get_default_encoding()

    # Create LeRobotDataset
    lerobot_dataset = LeRobotDataset.from_preloaded(
        repo_id="local/dataset",
        hf_dataset=hf_dataset,
        episode_data_index=episode_data_index,
        info=info,
        videos_dir=output_dir / "videos",
    )

    if run_compute_stats:
        logging.info("Computing dataset statistics")
        stats = compute_stats(lerobot_dataset)
        lerobot_dataset.stats = stats
    else:
        stats = {}
        logging.info("Skipping computation of the dataset statistics")

    # Save dataset components
    hf_dataset = hf_dataset.with_format(None)  # to remove transforms that can't be saved
    hf_dataset.save_to_disk(str(output_dir / "train"))

    meta_data_dir = output_dir / "meta_data"
    save_meta_data(info, stats, episode_data_index, meta_data_dir)

    print(f"Dataset converted and saved to {output_dir}")
    return lerobot_dataset


@hydra.main(version_base=None, config_path="../configs", config_name="to_lerobot")
def main(cfg: DictConfig) -> None:
    convert_to_lerobot_dataset(
        cfg.input_dir,
        cfg.output_dir,
        fps=cfg.dataset.fps,
        video=cfg.dataset.video,
        run_compute_stats=cfg.dataset.run_compute_stats)

if __name__ == "__main__":
    main()
