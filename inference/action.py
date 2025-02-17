import torch
import geom


def get_representation_size(representation: str) -> int:
    if representation == 'quat':
        return 4
    elif representation == 'euler':
        return 3
    elif representation == 'rotation_matrix':
        return 9
    elif representation == 'rotvec':
        return 3
    else:
        raise ValueError(f"Invalid rotation representation: {representation}")


def convert_quat_to(q: geom.Quaternion, representation: str) -> torch.Tensor:
    array = None
    if representation == 'quat':
        array = q
    elif representation == 'euler':
        array = q.as_euler
    elif representation == 'rotation_matrix':
        array = q.as_rotation_matrix
    elif representation == 'rotvec':
        array = q.as_rotvec
    else:
        raise ValueError(f"Invalid rotation representation: {representation}")

    return torch.from_numpy(array).flatten()


def convert_to_quat(array: torch.Tensor, representation: str) -> geom.Quaternion:
    if representation == 'quat':
        return geom.Quaternion(*array)
    elif representation == 'euler':
        return geom.Quaternion.from_euler(*array)
    elif representation == 'rotation_matrix':
        array = array.reshape(3, 3)
        return geom.Quaternion.from_rotation_matrix(array)
    elif representation == 'rotvec':
        return geom.Quaternion.from_rotvec(array)
    else:
        raise ValueError(f"Invalid rotation representation: {representation}")


class AbsolutePositionAction:
    def __init__(self, rotation_representation: str = 'quat'):
        self.rotation_representation = rotation_representation
        self.rotation_size = get_representation_size(rotation_representation)

    def encode_episode(self, episode_data):
        rotations = torch.zeros(len(episode_data['target_robot_position_quaternion']), self.rotation_size)

        # TODO: make this vectorized
        for i, q in enumerate(episode_data['target_robot_position_quaternion']):
            q = geom.Quaternion(*q)
            rotation = convert_quat_to(q, self.rotation_representation)
            rotations[i] = rotation

        translations = episode_data['target_robot_position_translation']
        grips = episode_data['target_grip']
        if grips.ndim == 1:
            grips = grips.unsqueeze(1)

        return torch.cat([rotations, translations, grips], dim=1)

    def decode(self, action_vector, inputs):
        mtx = action_vector[:self.rotation_size]
        q = convert_to_quat(mtx, self.rotation_representation)

        outputs = {
            'target_robot_position': geom.Transform3D(
                translation=action_vector[self.rotation_size:self.rotation_size + 3],
                quaternion=q
            ),
            'target_grip': action_vector[self.rotation_size + 3]
        }
        return outputs


class RelativeTargetPositionAction:
    def __init__(self, rotation_representation: str = 'quat'):
        self.rotation_representation = rotation_representation
        self.rotation_size = get_representation_size(rotation_representation)

    def encode_episode(self, episode_data):
        mtxs = torch.zeros(len(episode_data['target_robot_position_quaternion']), self.rotation_size)

        # TODO: make this vectorized
        for i, q_target in enumerate(episode_data['target_robot_position_quaternion']):
            q_target = geom.Quaternion(*q_target)
            q_current = geom.Quaternion(*episode_data['robot_position_quaternion'][i])
            q_relative = geom.quat_mul(q_current.inv, q_target)
            q_relative = geom.normalise_quat(q_relative)

            mtx = convert_quat_to(q_relative, self.rotation_representation)
            mtxs[i] = mtx.flatten()

        translation_diff = episode_data['target_robot_position_translation'] - episode_data['robot_position_translation']

        grips = episode_data['target_grip']
        if grips.ndim == 1:
            grips = grips.unsqueeze(1)

        return torch.cat([mtxs, translation_diff, grips], dim=1)

    def decode(self, action_vector, inputs):
        mtx = action_vector[:self.rotation_size]
        q_diff = convert_to_quat(mtx, self.rotation_representation)
        tr_diff = action_vector[self.rotation_size:self.rotation_size + 3]

        q_mul = geom.quat_mul(inputs['robot_position_quaternion'], q_diff)
        q_mul = geom.normalise_quat(q_mul)

        tr_add = inputs['robot_position_translation'] + tr_diff

        outputs = {
            'target_robot_position': geom.Transform3D(
                translation=tr_add,
                quaternion=q_mul
            ),
            'target_grip': action_vector[self.rotation_size + 3]
        }
        return outputs


class RelativeRobotPositionAction:
    def __init__(self, offset: int, rotation_representation: str = 'quat'):
        self.offset = offset
        self.rotation_representation = rotation_representation
        self.rotation_size = get_representation_size(rotation_representation)

    def encode_episode(self, episode_data):
        rotations = torch.zeros(len(episode_data['robot_position_quaternion']), self.rotation_size)
        translation_diff = -episode_data['robot_position_translation'].clone()
        grips = torch.zeros_like(episode_data['grip'])

        # TODO: make this vectorized
        for i, q_current in enumerate(episode_data['robot_position_quaternion']):
            if i + self.offset >= len(episode_data['robot_position_quaternion']):
                rotations[i] = convert_quat_to(geom.Quaternion(1, 0, 0, 0), self.rotation_representation)
                translation_diff[i] = torch.zeros(3)
                continue
            q_current = geom.Quaternion(*q_current)
            q_target = geom.Quaternion(*episode_data['robot_position_quaternion'][i + self.offset])
            q_relative = geom.quat_mul(q_current.inv, q_target)
            q_relative = geom.normalise_quat(q_relative)

            rotation = convert_quat_to(q_relative, self.rotation_representation)
            rotations[i] = rotation
            translation_diff[i] += episode_data['robot_position_translation'][i + self.offset]
            grips[i] = episode_data['grip'][i + self.offset]

        if grips.ndim == 1:
            grips = grips.unsqueeze(1)

        return torch.cat([rotations, translation_diff, grips], dim=1)

    def decode(self, action_vector, inputs):
        mtx = action_vector[:self.rotation_size]
        q_diff = convert_to_quat(mtx, self.rotation_representation)
        tr_diff = action_vector[self.rotation_size:self.rotation_size + 3]

        q_mul = geom.quat_mul(inputs['robot_position_quaternion'], q_diff)
        q_mul = geom.normalise_quat(q_mul)

        tr_add = inputs['robot_position_translation'] + tr_diff

        outputs = {
            'target_robot_position': geom.Transform3D(
                translation=tr_add,
                quaternion=q_mul
            ),
            'target_grip': action_vector[self.rotation_size + 3]
        }
        return outputs
