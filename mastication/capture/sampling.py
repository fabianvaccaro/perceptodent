import numpy as np
from PIL import Image

class Gum(object):
    """Chewing gum sample, containing both the image and the masticatory assessment metrics"""
    def __init__(self, rgb, t_value):
        assert isinstance(rgb, np.ndarray)
        self.rgb = rgb
        self.t_value = t_value


def FromImage(pth, t_value):
    image = Image.open(pth)
    pix = np.array(image.getdata())
    non_alpha = pix[:, [0, 1, 2]]
    non_alpha = non_alpha.reshape(image.size[1], image.size[0], 3)
    # image = np.array(image)
    # non_alpha = image[:,:,[0,1,2]]
    return Gum(non_alpha, t_value)
