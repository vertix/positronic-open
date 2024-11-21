# It is essential to use driverless disk image on Nebius to make below installation work

# Update and install CUDA and drivers
sudo apt-get update
sudo apt-get install -y build-essential gcc make
sudo apt-get install -y cuda-toolkit-11-8 nvidia-driver-555
sudo apt-get install -y libcudnn8=8.6.0.*-1+cuda11.8 libcudnn8-dev=8.6.0.*-1+cuda11.8

# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
chmod +x ~/miniconda.sh
~/miniconda.sh -b -p ~/miniconda
eval "$($HOME/miniconda/bin/conda shell.bash hook)"
echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc

# Install Octo
git clone https://github.com/vertix/octo.git
cd octo
conda create -y -n octo python=3.10
conda activate octo
pip install -e .
pip install -r requirements.txt

pip install scipy==1.11.3
pip install --upgrade "jax[cuda11_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Download example dataset
cd ..
wget https://rail.eecs.berkeley.edu/datasets/example_sim_data.zip
sudo apt install -y unzip
unzip example_sim_data.zip

# Install Act
git clone https://github.com/tonyzhaozh/act.git
pip install opencv-python modern_robotics pyrealsense2 h5py_cache pyquaternion pyyaml rospkg pexpect mujoco==2.3.3 dm_control==1.0.9 einops packaging h5py
pip install jupyterlab

pip install wandb
wandb login

# Restart before running finetuning
sudo reboot

# Run finetuning
cd octo
PYTHONPATH=$PYTHONPATH:$(pwd)/../act python examples/02_finetune_new_observation_action.py --pretrained_path=hf://rail-berkeley/octo-small-1.5 --data_dir=../aloha_sim_dataset --save_dir=$(pwd)/../octo_finetuning

sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt-get update

# Install newer GCC and libstdc++
sudo apt-get install -y gcc-11 g++-11 libstdc++6

# Run evaluation
export DISPLAY=:0
Xvfb :0 -screen 0 1024x768x24 &
sleep 2

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
PYTHONPATH=$PYTHONPATH:$(pwd)/../act python examples/03_eval_finetuned.py --finetuned_path=$(pwd)/../octo_finetuning/
