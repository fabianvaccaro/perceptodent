import numpy as np


class Gum(object):
    """Chewing gum sample, containing both the image and the masticatory assessment metrics"""
    def __init__(self, rgb, me, mp):
        assert isinstance(rgb, np.ndarray)
        assert isinstance(me, float)
        assert isinstance(mp, np.ndarray)
        self.rgb = rgb
        self.mp = mp
        self.me = me
