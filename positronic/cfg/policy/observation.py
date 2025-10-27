import configuronic as cfn

from positronic.policy.observation import ObservationEncoder


@cfn.config(image_size=(224, 224))
def eepose_grip(wrist_camera: str, side_camera: str, image_size: tuple[int, int]):
    state = {'observation.state': ['robot_state.ee_pose', 'grip']}
    state_dim = 8
    images = {
        'observation.images.left': (wrist_camera, image_size),
        'observation.images.side': (side_camera, image_size),
    }

    result = ObservationEncoder(state=state, images=images)
    result.meta['gr00t_modality'] = {
        'state': {
            'robot_position_quaternion': {'start': 0, 'end': 4, 'rotation_type': 'quaternion'},
            'robot_position_translation': {'start': 4, 'end': 7},
            'grip': {'start': 7, 'end': 8},
        },
        'video': {
            'ego_view': {'original_key': 'observation.images.left'},
            'side_image': {'original_key': 'observation.images.side'},
        },
    }
    lerobot_features = {k: {'shape': (state_dim,), 'names': v, 'dtype': 'float32'} for k, v in state.items()}
    for out_name, (_input_key, (width, height)) in result._image_configs.items():
        lerobot_features[out_name] = {
            'shape': (height, width, 3),
            'names': ['height', 'width', 'channel'],
            'dtype': 'video',
        }
    result.meta['lerobot_features'] = lerobot_features
    return result


eepose_mujoco = eepose_grip.override(wrist_camera='image.handcam_left', side_camera='image.back_view')
eepose_real = eepose_grip.override(wrist_camera='image.wrist', side_camera='image.exterior')


@cfn.config(exterior_camera='image.exterior', wrist_camera='image.wrist', image_size=(224, 224))
def openpi_positronic(exterior_camera: str, wrist_camera: str, image_size: tuple[int, int]):
    return ObservationEncoder(
        state={'observation/state': ['robot_state.ee_pose', 'grip']},
        images={
            'observation/wrist_image': (wrist_camera, image_size),
            'observation/image': (exterior_camera, image_size),
        },
    )


@cfn.config(exterior_camera='image.exterior', wrist_camera='image.wrist', image_size=(224, 224))
def openpi_droid(exterior_camera: str, wrist_camera: str, image_size: tuple[int, int]):
    """DROID observation encoder using joint positions."""
    return ObservationEncoder(
        state={'observation/joint_position': ['robot_state.q'], 'observation/gripper_position': ['grip']},
        images={
            'observation/wrist_image_left': (wrist_camera, image_size),
            'observation/exterior_image_1_left': (exterior_camera, image_size),
        },
    )


@cfn.config()
def groot_oxe_droid():
    state_spec = {'observation.state': ['robot_state.ee_pose', 'grip']}
    result = ObservationEncoder(
        state=state_spec,
        images={
            'observation.images.exterior': ('image.exterior', (224, 224)),
            'observation.images.wrist': ('image.wrist', (224, 224)),
        },
    )
    result.meta['gr00t_modality'] = {
        'state': {
            'eef_rotation': {'start': 0, 'end': 4, 'rotation_type': 'quaternion'},
            'eef_position': {'start': 4, 'end': 7},
            'gripper_position': {'start': 7, 'end': 8},
        },
        'video': {
            'exterior_image_1': {'original_key': 'observation.images.exterior'},
            'exterior_image_2': {'original_key': 'observation.images.exterior'},
            'wrist_image': {'original_key': 'observation.images.wrist'},
        },
    }
    state_dim = 8
    lerobot_features = {k: {'shape': (state_dim,), 'names': v, 'dtype': 'float32'} for k, v in state_spec.items()}
    for out_name, (_input_key, (width, height)) in result._image_configs.items():
        lerobot_features[out_name] = {
            'shape': (height, width, 3),
            'names': ['height', 'width', 'channel'],
            'dtype': 'video',
        }
    result.meta['lerobot_features'] = lerobot_features
    return result


@cfn.config()
def groot_infer():
    state_spec = {'observation.state': ['robot_state.ee_pose', 'grip']}
    result = ObservationEncoder(
        state=state_spec,
        images={
            'video.exterior_image_1': ('image.exterior', (224, 224)),
            'video.exterior_image_2': ('image.exterior', (224, 224)),
            'video.wrist_image': ('image.wrist', (224, 224)),
        },
    )
    return result
