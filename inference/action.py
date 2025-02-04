import torch
import geom


class AbsolutePositionAction:
    def encode_episode(self, episode_data):
        rotations = torch.zeros(len(episode_data['target_robot_position_quaternion']), 9)

        # TODO: make this vectorized
        for i, q in enumerate(episode_data['target_robot_position_quaternion']):
            q = geom.Quaternion(*q)
            mtx = q.as_rotation_matrix
            rotations[i] = torch.from_numpy(mtx.flatten())


        translations = episode_data['target_robot_position_translation']
        grips = episode_data['target_grip']
        if grips.ndim == 1:
            grips = grips.unsqueeze(1)

        return torch.cat([rotations, translations, grips], dim=1)

    def decode(self, action_vector, inputs):
        mtx = action_vector[:9].reshape(3, 3)
        q = geom.Quaternion.from_rotation_matrix(mtx)
        outputs = {
            'target_robot_position': geom.Transform3D(
                translation=action_vector[9:12],
                quaternion=q
            ),
            'target_grip': action_vector[12]
        }
        return outputs


class RelativePositionAction:
    def __init__(self):
        pass

    def encode_episode(self, episode_data):
        mtxs = torch.zeros(len(episode_data['target_robot_position_quaternion']), 9)

        # TODO: make this vectorized
        for i, q in enumerate(episode_data['target_robot_position_quaternion']):
            q = geom.Quaternion(*q)
            q_inv = geom.Quaternion(*episode_data['robot_position_quaternion'][i]).inv
            q_mul = geom.quat_mul(q_inv, q)
            q_mul = geom.normalise_quat(q_mul)

            mtx = q_mul.as_rotation_matrix
            mtxs[i] = torch.from_numpy(mtx.flatten())

        translation_diff = episode_data['target_robot_position_translation'] - episode_data['robot_position_translation']

        grips = episode_data['target_grip']
        if grips.ndim == 1:
            grips = grips.unsqueeze(1)

        return torch.cat([mtxs, translation_diff, grips], dim=1)

    def decode(self, action_vector, inputs):
        mtx = action_vector[:9].reshape(3, 3)
        q_diff = geom.Quaternion.from_rotation_matrix(mtx)
        tr_diff = action_vector[9:12]

        q_mul = geom.quat_mul(inputs['robot_position_quaternion'], q_diff)
        q_mul = geom.normalise_quat(q_mul)

        tr_add = inputs['robot_position_translation'] + tr_diff

        outputs = {
            'target_robot_position': geom.Transform3D(
                translation=tr_add,
                quaternion=q_mul
            ),
            'target_grip': action_vector[12]
        }
        return outputs
