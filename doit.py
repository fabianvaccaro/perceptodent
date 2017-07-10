from mastication.assessment import calibration
import matplotlib.pyplot as plt
import numpy as np

if(__name__=='__main__'):
    results = calibration.calibrate_samples()
    print(results)
    # plt.matshow(mfc_matrix)
    # plt.show()
