import numpy as np
from mastication.characterization import MFC
from mastication.segmentation import segmenter
from skimage import color
from mastication.capture import sampling


def cmfc(sampleimage):
    assert isinstance(sampleimage, sampling.Gum)
    ms = segmenter.ChewingGumSegmenter()
    r = ms.segment(sampleimage)
    roi_val = r[int(r.shape[0] / 2), int(r.shape[1] / 2)]
    rgb = sampleimage.rgb[r == roi_val]
    luv = color.rgb2luv(sampleimage.rgb)[r == roi_val]
    hsi = color.rgb2hsv(sampleimage.rgb)[r == roi_val]
    nrgb = MFC.rgb2nrgb(sampleimage.rgb)[r == roi_val]
    mfc = MFC.extractmfc(rgb, luv, hsi, nrgb)
    return mfc