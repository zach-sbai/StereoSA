from .StereoSA import StereoSA
from .StereoSA_trt import StereoSA_trt
from .StereoSA_confidence import StereoSA_confidence

from .loss import model_loss_train, model_loss_test

__models__ = {
    "StereoSA": StereoSA,
    "StereoSA_trt": StereoSA_trt,
    "StereoSA_confidence": StereoSA_confidence
}
