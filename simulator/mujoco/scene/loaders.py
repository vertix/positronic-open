import abc
import os
from typing import Any, Dict, Sequence
import xml.etree.ElementTree as ET


def add_camera(root: ET.Element, name: str, pos: str, xyaxes: str):
    camera = ET.Element("camera", name=name, mode="fixed", pos=pos, xyaxes=xyaxes)
    worldbody = root.find("worldbody")
    worldbody.append(camera)


class MujocoSceneLoader(abc.ABC):
    @abc.abstractmethod
    def apply(self, root: ET.Element) -> ET.Element:
        pass


class AddCameras(MujocoSceneLoader):
    def __init__(self, additional_cameras: Dict[str, Dict[str, Any]]):
        self.additional_cameras = additional_cameras

    def apply(self, root: ET.Element) -> ET.Element:

        for camera_name, camera_cfg in self.additional_cameras.items():
            add_camera(root, camera_name, camera_cfg.pos, camera_cfg.xyaxes)

        return root
    
class ReplaceAssetPaths(MujocoSceneLoader):
    def __init__(self, orig_path: str, new_path: str) -> None:
        super().__init__()
        self.orig_path = orig_path
        self.new_path = new_path

    def apply(self, root: ET.Element) -> ET.Element:
        for item in root.iter():
            if item.attrib.get("file"):
                item.attrib["file"] = item.attrib["file"].replace(self.orig_path, self.new_path)
        return root
    
class FixRelativePaths(MujocoSceneLoader):
    def __init__(self, base_dir: str) -> None:
        super().__init__()
        print(base_dir)
        self.base_dir = base_dir

    def apply(self, root: ET.Element) -> ET.Element:
        for item in root.iter():
            if item.attrib.get("file"):
                item.attrib["file"] = os.path.join(self.base_dir, item.attrib["file"])
        return root
    
class RecolorObject(MujocoSceneLoader):
    def __init__(self, object_name: str, color: str) -> None:
        super().__init__()
        self.object_name = object_name
        self.color = color

    def apply(self, root: ET.Element) -> ET.Element:
        # geom with name self.object_name
        geom = root.find(f".//geom[@name='{self.object_name}']")
        if geom is not None:
            geom.attrib["rgba"] = self.color
        return root


def load_xml_string(xml_string: str, loaders: Sequence[MujocoSceneLoader] = ()) -> str:
    tree = ET.ElementTree(ET.fromstring(xml_string))
    root = tree.getroot()

    for loader in loaders:
        root = loader.apply(root)

    return ET.tostring(root)
