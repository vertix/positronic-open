##
Clone repo

## Install conda


## Setup conda
```python
conda env create --file environment.yaml --name positronic
conda activate positronic
```

## Install eigen3 and libfranka
```bash
sudo apt install libeigen3-dev
git clone --recursive https://github.com/frankaemika/libfranka
cd libfranka
git checkout 0.14.1
git submodule update

mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF ..
cmake --build .
```

## Install franky
Clone franky into parent directory
```bash
git clone --branch v0.9.1 git@github.com:TimSchneider42/franky.git
cd franky
pip install -e .
```

## How to convert teleoperated dataset into LeRobot format (ready for training)
```python
python training/to_lerobot.py output_dir=_lerobot_ds/
```

By default, this reads data from `_dataset` directory. Use `input_dir=your_dir` to control inputs. Please refer to [configs/to_lerobot.yaml](../configs/to_lerobot.yaml) for more details.