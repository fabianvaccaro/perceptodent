import numpy as np
from sklearn.cluster import  mean_shift, k_means, estimate_bandwidth
from scipy import ndimage
import matplotlib
from mastication.capture import sampling


class ISegmenter(object):
    def __init__(self):
        pass

    def _segment(self, sampleimage):
        raise NotImplementedError()

    def segment(self, sampleimage):
        assert isinstance(sampleimage, sampling.Gum)
        res = self._segment(sampleimage)
        assert isinstance(res, np.ndarray)
        return res


class MeanShift(ISegmenter):
    def _segment(self, sampleimage):
        flat_image = np.reshape(sampleimage, [-1, 3])






