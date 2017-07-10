import numpy as np
from scipy.signal import find_peaks_cwt


def meanofpixels(maskedimage):
    assert isinstance(maskedimage, np.ndarray)
    return np.mean(maskedimage)


def varianceofpixels(maskedimage):
    assert isinstance(maskedimage, np.ndarray)
    mu = meanofpixels(maskedimage)
    A = maskedimage.shape[0]
    V = 0
    for y in range(0, A):
        V = V + np.abs(maskedimage[y] - mu)
    V = V / (A - 1)
    return V


def _gethistogram(maskedimage):
    assert isinstance(maskedimage, np.ndarray)
    return np.histogram(maskedimage, bins=100)[0]


def _meanofhistogram(maskedimage):
    assert isinstance(maskedimage, np.ndarray)
    ndhist = _gethistogram(maskedimage)
    return np.mean(ndhist)


def varianceofhistogram(maskedimage):
    assert isinstance(maskedimage, np.ndarray)
    ndhist = _gethistogram(maskedimage)
    mh = _meanofhistogram(maskedimage)
    Vh = 0
    for x in range(0, 100):
        Vh = Vh + np.abs(ndhist[x] - mh)
    Vh = Vh / 99
    return Vh


def skewnessofhistogram(maskedimage):
    assert isinstance(maskedimage, np.ndarray)
    ndhist = _gethistogram(maskedimage)
    Vh = varianceofhistogram(maskedimage)
    mh = _meanofhistogram(maskedimage)
    Sh = 0
    for x in range(0, 100):
        Sh = Sh + ((maskedimage[x] - mh)*(maskedimage[x] - mh)*(maskedimage[x] - mh))/(99*Vh*Vh*Vh)
    return Sh


def energyofhistogram(maskedimage):
    assert isinstance(maskedimage, np.ndarray)
    ndhist = _gethistogram(maskedimage)
    Eh = 0
    for x in range(0, 100):
        Eh = Eh + (ndhist[x] * ndhist[x])
    return Eh


def entropyofhistogram(maskedimage):
    assert isinstance(maskedimage, np.ndarray)
    ndhist = _gethistogram(maskedimage)
    Nh = 0
    for x in range(0, 100):
        if ndhist[x] > 0:
            Nh = Nh + ndhist[x] * np.log2(ndhist[x])
    Nh = -Nh
    return Nh


def circularvariance(maskedimage):
    assert isinstance(maskedimage, np.ndarray)
    A = maskedimage.shape[0]
    pt1 = 0
    pt2 = 0
    for x in range(0, A):
        pt1 = pt1 + np.cos(maskedimage[x])
        pt2 = pt2 + np.sin(maskedimage[x])
    pt1 = pt1 * pt1
    pt2 = pt2 * pt2
    cvoh = 1 - (1/A)*(np.sqrt(pt1+pt2))
    return cvoh

def _findpeaks(ndhist):
    return find_peaks_cwt(ndhist, np.arange(1, 10))



def higestpeakvalue(maskedimage):
    assert isinstance(maskedimage, np.ndarray)
    ndhist = _gethistogram(maskedimage)
    peaks = _findpeaks(ndhist)
    if(len(peaks) >= 1):
        return ndhist[peaks[0]]
    else:
        return max(ndhist)


def secondhigestpeakvalue(maskedimage):
    assert isinstance(maskedimage, np.ndarray)
    ndhist = _gethistogram(maskedimage)
    peaks = _findpeaks(ndhist)
    if(len(peaks) >= 2):
        return ndhist[peaks[1]]
    else:
        return max(ndhist)


def higestpeakposition(maskedimage):
    assert isinstance(maskedimage, np.ndarray)
    ndhist = _gethistogram(maskedimage)
    peaks = _findpeaks(ndhist)
    if(len(peaks) >= 1):
        return peaks[0]
    else:
        i, = np.where(ndhist==np.max(ndhist))
        return i[0]


def secondhigestpeakposition(maskedimage):
    assert isinstance(maskedimage, np.ndarray)
    ndhist = _gethistogram(maskedimage)
    peaks = _findpeaks(ndhist)
    if(len(peaks) >= 2):
        return peaks[1]
    else:
        i, = np.where(ndhist==np.max(ndhist))
        return i[0]




