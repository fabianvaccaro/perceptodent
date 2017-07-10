import numpy as np
from mastication.capture import sampling
from sklearn.decomposition import PCA


class MEPAT(object):
    def __init__(self, classifier, pca):
        assert isinstance(pca, PCA)
        self._classifier = classifier
        self._pca = pca
        pass

    def ComputeME(self, sampleimage):
        assert isinstance(sampleimage, sampling.Gum)

    def ComputeMFC(self, sampleimage):
        assert isinstance(sampleimage, sampling.Gum)
