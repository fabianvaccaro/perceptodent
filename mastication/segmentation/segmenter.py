import numpy as np
from sklearn.cluster import  KMeans, estimate_bandwidth
from sklearn.cluster import  MeanShift as mmss
from scipy import ndimage
import matplotlib
from mastication.capture import sampling
from skimage import io, color

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
        assert isinstance(sampleimage, sampling.Gum)
        flat_image = np.reshape(sampleimage.rgb, [-1, 3])
        bandwidth2 = estimate_bandwidth(flat_image, quantile=.2, n_samples=500)
        ms = mmss(bandwidth2, bin_seeding=True)
        ms.fit(flat_image)
        labels = ms.labels_
        return np.array(np.reshape(labels, [sampleimage.rgb.shape[0], sampleimage.rgb.shape[1]]))


class ChewingGumSegmenter(ISegmenter):
    def _segment(self, sampleimage):
        # ms = MeanShift()
        # mask_ms = ms.segment(sampleimage)
        lab_b = color.rgb2lab(sampleimage.rgb)[:, :, 2]
        h = sampleimage.rgb.shape[0]
        w = sampleimage.rgb.shape[1]
        window_size = 40
        tl = lab_b[0:window_size, 0:window_size].flatten()
        tr = lab_b[0:window_size, w-1-window_size:w-1].flatten()
        bl = lab_b[h-1-window_size:h-1, 0:window_size].flatten()
        br = lab_b[h-1-window_size:h-1, w-1-window_size:w-1].flatten()
        mean_corner_colour = np.mean(np.concatenate((tl, tr, bl, br), axis=0))

        distance_map = np.zeros((h, w))
        for y in range(0,h):
            for x in range(0, w):
                distance_map[y, x] = np.log(np.abs(lab_b[y, x]-mean_corner_colour))
        feature_array = np.reshape(distance_map, (w*h,1))
        kmeans = KMeans(n_clusters=2, random_state=0).fit(feature_array)
        lbls = np.reshape(kmeans.predict(feature_array), (h, w))
        return lbls

