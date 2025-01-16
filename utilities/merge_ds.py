import os

import glob
import fire


def merge_ds(d1: str, d2: str):
    """
    Merge first dataset into second dataset.
    
    Dataset 1 has structure:
    - d1/
        - 001.pt
        - 002.pt
        - ...
    Dataset 2 has structure:
    - d2/
        - 001.pt
        - 002.pt
        - ...
    Merged dataset has structure:
    - d2/
        - 001.pt
        - 002.pt
        - ...
        - N.pt    # Below is the samples from dataset 1
        - N+1.pt
        - ...
    """
    d1_files = sorted(glob.glob(os.path.join(d1, "*.pt")))
    d2_files = sorted(glob.glob(os.path.join(d2, "*.pt")))

    if len(d2_files) == 0:
        last_d2_file_idx = 0
    else:
        last_d2_file = os.path.basename(d2_files[-1])
        last_d2_file_idx = int(last_d2_file.split(".")[0])

    for d1_file in d1_files:
        d1_file_idx = int(os.path.basename(d1_file).split(".")[0])
        new_d2_file = os.path.join(d2, f"{last_d2_file_idx + d1_file_idx:03d}.pt")
        os.symlink(d1_file, new_d2_file)

    assert len(glob.glob(os.path.join(d2, "*.pt"))) == len(d1_files) + len(d2_files)

if __name__ == "__main__":
    fire.Fire(merge_ds)
