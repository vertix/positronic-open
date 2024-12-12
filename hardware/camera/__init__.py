import abc
from typing import Dict, Tuple

import numpy as np


class Camera(abc.ABC):
    @abc.abstractmethod
    def setup(self):
        pass

    @abc.abstractmethod
    def cleanup(self):
        pass

    @abc.abstractmethod
    def get_frame(self) -> Tuple[Dict[str, np.ndarray], float]:
        pass
    