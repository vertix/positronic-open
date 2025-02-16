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


class RelativeTargetPositionAction:
    def encode_episode(self, episode_data):
        mtxs = torch.zeros(len(episode_data['target_robot_position_quaternion']), 9)

        # TODO: make this vectorized
        for i, q_target in enumerate(episode_data['target_robot_position_quaternion']):
            q_target = geom.Quaternion(*q_target)
            q_current = geom.Quaternion(*episode_data['robot_position_quaternion'][i])
            q_relative = geom.quat_mul(q_current.inv, q_target)
            q_relative = geom.normalise_quat(q_relative)

            mtx = q_relative.as_rotation_matrix
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


class RelativeRobotPositionAction:
    def __init__(self, offset: int):
        self.offset = offset

    def encode_episode(self, episode_data):
        mtxs = torch.zeros(len(episode_data['robot_position_quaternion']), 9)
        translation_diff = -episode_data['robot_position_translation'].clone()
        grips = torch.zeros_like(episode_data['grip'])

        # TODO: make this vectorized
        for i, q_current in enumerate(episode_data['robot_position_quaternion']):
            if i + self.offset >= len(episode_data['robot_position_quaternion']):
                mtxs[i] = torch.eye(3).flatten()
                translation_diff[i] = torch.zeros(3)
                continue
            q_current = geom.Quaternion(*q_current)
            q_target = geom.Quaternion(*episode_data['robot_position_quaternion'][i + self.offset])
            q_relative = geom.quat_mul(q_current.inv, q_target)
            q_relative = geom.normalise_quat(q_relative)

            mtx = q_relative.as_rotation_matrix
            mtxs[i] = torch.from_numpy(mtx.flatten())
            translation_diff[i] += episode_data['robot_position_translation'][i + self.offset]
            grips[i] = episode_data['grip'][i + self.offset]

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
