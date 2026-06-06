import numpy as np
import cv2 as cv

npzfile = np.load('calibration_data.npz')
sorted(npzfile.files)
print(npzfile['mtx']) # mtx