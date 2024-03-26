# AUTH: Andy Cox V
# DATE: 13 MAR 2024
# LANG: Python 3.10.7
# USAG: mat2npz.py (make sure that *.mat files are in same directory!)
# DESC: ECEN689 - Converts MATLAB data files into a format easily digestible by Python.
#       There are 60,000 8x8=64-bit images which will be loaded as a dictionary of labels and their corresponding data.
#       Use numpy to load the labels ('x') and data ('y').
#       Each label at index i corresponds to its data at index i.
#       Labels (dictionary key = 'x') at each index i are stored as a uint8.
#       Data (dictionary key = 'y') at index i is stored as a 64 element array of float32.
#       Example:
#           validate = np.load('MNIST_8x8.npz')
#           validate['x'][100] = 1 element of uint8 as label at location 100
#           validate['y'][100] = 64 elements of float32 data at location 100

import scipy.io
import numpy as np
from matplotlib import pyplot as plt

img = scipy.io.loadmat('MNIST_TrainSet_0to1_8x8pixel.mat')
lab = scipy.io.loadmat('MNIST_TrainSet_Label.mat')

# If wanting to combine label data and actual data into a single array need to transpose instead of flatten and turn uint8 to float32.
# Get label data.
labData = lab['label'].flatten().astype(np.uint8)  # uint8 and flatten from 1x60000 to just 60000.

# Get image data.
imgData = np.ndarray((64, 60000), np.float32) # Each image is 64-bits and there are 60,000 samples.
row = 0
for value in img['number']:  # Each value is a row of 60,000. The numbers are stored as columns.
    np.copyto(imgData[row], value, casting='same_kind')
    row += 1
imgData = imgData.transpose()

# Write to data to compressed npz file where x = label data and y = image data.
np.savez_compressed('MNIST_8x8.npz', x=labData, y=imgData)

# Validate saved data.
validate = np.load('MNIST_8x8.npz')
valLables = validate['x']
valData = validate['y']
testRow = 100

# Printouts.
print('Validating Data')
print(f'Labels Shape = {valLables.shape}')
print(f'Data Shape = {valData.shape}')
print(f'Test Row={testRow}: Label = {valLables[testRow]}, Data =', valData[testRow])

# Plot test data.
plt.title(f'Test Row = {testRow}: Data Plot Below Should = {valLables[testRow]}')
plt.imshow(np.reshape(valData[testRow], (8, 8)), interpolation='none', cmap='gray') # Need to reshape into 8x8 image.
plt.show()
exit()