## How to convert teleoperated dataset into LeRobot format (ready for training)
```python
python training/to_lerobot.py output_dir=_lerobot_ds/
```

By default, this reads data from `_dataset` directory. Use `input_dir=your_dir` to control inputs. Please refer to [configs/to_lerobot.yaml](../configs/to_lerobot.yaml) for more details.