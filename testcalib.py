from mastication.assessment import calibration
import matplotlib.pyplot as plt
import numpy as np

if(__name__=='__main__'):
    results = calibration.test_calibration()
    print(results)
