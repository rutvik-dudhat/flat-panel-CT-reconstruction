import numpy as np
import Methods as Methods
from EX_1.Grid import Grid

from EX_1.phantom import phantom
import time
import matplotlib.pyplot as plt
#from interpolate import interpolate as ip
from EX_1.interpolate import interpolate as ip
from scipy import interpolate


shepp = phantom(n=64, p_type='Modified Shepp-Logan', ellipses=None)
grid_size = [64, 64]
grid_spacing = [0.5, 0.5]
sheppGrid = Grid(grid_size[0], grid_size[1], grid_spacing)
sheppGrid.set_buffer(shepp)


# sinogram parameters
number_of_projections = 180
detector_spacing = 0.5
detector_size = 96
scan_range_in_degree = 180

# create sinogram
t0 = time.time()
#sinogram = Grid(detector_size, 1, (detector_spacing, detector_spacing))
#sinogram = Grid(detector_size, 1, number_of_projections, (0,0),detector_spacing)
sinogram = Methods.create_sinogram(sheppGrid, number_of_projections, detector_spacing, detector_size, scan_range_in_degree)
t1 = time.time()
print('sinogram:', t1-t0)



plt.imshow(sinogram.buffer)
plt.gray()
plt.show()

plt.figure()
t0 = time.time()
reco = Methods.backproject(sinogram, grid_size[0], grid_size[1], grid_spacing)
t1 = time.time()
print('backprojection:', t1-t0)
plt.imshow(reco.buffer)
plt.gray()
plt.show()

# filter sinogram
t0 = time.time()
sinogram_filtered = Methods.ramp_filter(sinogram, sinogram.get_spacing()[1])
#sinogram_filtered=Methods.ramlak_filter(sinogram, sinogram.get_spacing()[1])
#sinogram_filtered = sinogram_filtered.reshape((90, 90))
t1 = time.time()
print('ramp filter:', t1-t0)
plt.imshow(sinogram_filtered.buffer)
plt.gray()
plt.show()

# create filtered reco
t0 = time.time()
reco_filtered = Methods.backproject(sinogram_filtered, grid_size[0], grid_size[1], grid_spacing)
t1 = time.time()
print('backprojection:', t1-t0)
plt.imshow(reco_filtered.buffer)
plt.gray()
plt.show()

