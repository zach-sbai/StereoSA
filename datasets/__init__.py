from .kitti_dataset import KITTIDataset
from .vkitti_dataset import vKITTIDataset
from .sceneflow_dataset import SceneFlowDatset

__datasets__ = {
    "sceneflow": SceneFlowDatset,
    "kitti": KITTIDataset,
    "vkitti": vKITTIDataset
}
