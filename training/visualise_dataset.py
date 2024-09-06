import click
import torch

from control import utils
from control.world import MainThreadWorld
import geom
from tools.rerun import Rerun, log_image

def source_control_system(dataset_path: str):
    dataset = torch.load(dataset_path)
    output_keys = set(dataset.keys()) - {"time", 'time/now', 'time/robot', 'time/image'}
    output_keys.add('robot_position')
    output_keys.update({'robot_position',})

    @utils.control_system(inputs=[], outputs=output_keys)
    def _generator_system(_ins, outs):
        for i in range(len(dataset['time'])):
            ts = dataset['time'][i] * 1000
            for key in output_keys:
                if key == 'robot_position':
                    pos = geom.Transform3D(translation=dataset['robot_position_trans'][i].numpy(),
                                           quaternion=dataset['robot_position_quat'][i].numpy())
                    outs.robot_position.write(pos, ts)
                else:
                    outs[key].write(dataset[key][i].numpy(), ts)

    return _generator_system


@click.command()
@click.argument("dataset_path", type=click.Path(exists=True))
@click.option("--rerun", type=str, required=True,
              help="Either path to rerun server like '127.0.0.1:9876' or a path to dump file like 'dump.rrd'")
def visualise_dataset(dataset_path: str, rerun: str):
    """Visualise a dataset from a file."""
    world = MainThreadWorld()
    feeder = source_control_system(dataset_path)(world)

    inputs = {o: None for o in feeder.outs.available_ports()}

    if ":" in rerun:
        rr = Rerun(world, "dataset", connect=rerun, inputs=inputs)
    else:
        rr = Rerun(world, "dataset", save_path=rerun, inputs=inputs)

    for key in feeder.outs.available_ports():
        rr.ins[key] = feeder.outs[key]

    world.run()


if __name__ == "__main__":
    visualise_dataset()