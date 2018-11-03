import numpy as np
from mastication.characterization import features


def extractmfc(rgb, luv, hsi, nrgb):
    colour_spaces = {
        'rgb': rgb,
        'luv': luv,
        'hsi': hsi,
        'nrgb': nrgb
    }
    mfc = np.zeros(121)
    cnt = 0
    for cspace in colour_spaces:
        for channel in range(0, 3):
            valuearray = colour_spaces[cspace][:, channel]
            mfc[cnt] = features.meanofpixels(valuearray)
            cnt += 1
            mfc[cnt] = features.varianceofpixels(valuearray)
            cnt += 1
            mfc[cnt] = features.higestpeakposition(valuearray)
            cnt += 1
            mfc[cnt] = features.secondhigestpeakposition(valuearray)
            cnt += 1
            mfc[cnt] = features.higestpeakvalue(valuearray)
            cnt += 1
            mfc[cnt] = features.secondhigestpeakvalue(valuearray)
            cnt += 1
            mfc[cnt] = features.varianceofhistogram(valuearray)
            cnt += 1
            mfc[cnt] = features.skewnessofhistogram(valuearray)
            cnt += 1
            mfc[cnt] = features.energyofhistogram(valuearray)
            cnt += 1
            mfc[cnt] = features.entropyofhistogram(valuearray)
            cnt += 1
    mfc[cnt] = features.entropyofhistogram(colour_spaces['hsi'][:, 0])
    return mfc


def rgb2nrgb(rgb):
    assert isinstance(rgb, np.ndarray)
    nrgb = np.zeros((rgb.shape[0], rgb.shape[1], 3))
    for y in range(0, rgb.shape[0]):
        for x in range(0, rgb.shape[1]):
            total = rgb[y, x, 0] + rgb[y, x, 1] + rgb[y, x, 2]
            if total == 0:
                nrgb[y, x, 0] = 0
                nrgb[y, x, 1] = 0
                nrgb[y, x, 2] = 0
            else:
                nrgb[y, x, 0] = np.round((rgb[y, x, 0] / total) * 255, decimals=0)
                nrgb[y, x, 1] = np.round((rgb[y, x, 1] / total) * 255, decimals=0)
                nrgb[y, x, 2] = np.round((rgb[y, x, 2] / total) * 255, decimals=0)
    return nrgb






