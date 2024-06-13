#import Methods as Methods
from EX_1.Grid import Grid
from EX_1.phantom import phantom
import time
import matplotlib.pyplot as plt
import Mode as Methods



shepp = phantom(n=64, p_type='Modified Shepp-Logan', ellipses=None)
grid_size = [64, 64]
grid_spacing = [0.5, 0.5]
sheppGrid = Grid(grid_size[0], grid_size[1], grid_spacing)
sheppGrid.set_buffer(shepp)


##Reconstruction from rebinning ####################################

# fanogram parameters
detector_spacing = 0.5
detector_size = sheppGrid.height + sheppGrid.width
number_of_projections = 180
scan_range = 220
angular_increment = scan_range / number_of_projections
d_si = 700
d_sd = 1200

# create fanogram
t0 = time.time()
fanogram_shortScan = \
Methods.create_fanogram(sheppGrid, number_of_projections, detector_spacing, detector_size, angular_increment, d_si, d_sd)
t1 = time.time()
print('fanogram:', t1-t0)

plt.imshow(fanogram_shortScan.buffer)
plt.gray()
plt.show()

sinogram = Methods.rebinning(fanogram_shortScan, d_si, d_sd)
sinogram_filtered = Methods.ramp_filter(sinogram, sinogram.get_spacing()[1])
reco_from_sinogram = Methods.backproject(sinogram_filtered, grid_size[0], grid_size[1], grid_spacing)
plt.imshow(reco_from_sinogram.buffer)
plt.gray()
plt.show()
## Direct reconstruction ############################################

# fanogram parameters
detector_spacing = 0.5
detector_size = sheppGrid.height + sheppGrid.width
number_of_projections = 180
scan_range = 360
angular_increment = scan_range / number_of_projections
d_si = 700
d_sd = 1200

# create fanogram
t0 = time.time()
fanogram_fullScan = \
Methods.create_fanogram(sheppGrid, number_of_projections, detector_spacing, detector_size, angular_increment, d_si, d_sd)
t1 = time.time()
print('fanogram:', t1-t0)

plt.imshow(fanogram_fullScan.buffer)
plt.gray()
plt.show()


# filter sinogram
t0 = time.time()
fanogram_filtered = Methods.ramp_filter(fanogram_fullScan, fanogram_fullScan.get_spacing()[1])
t1 = time.time()
print('ramp filter:', t1-t0)

# create filtered reco
t0 = time.time()
reco_filtered = Methods.backproject_fanbeam(fanogram_filtered, grid_size[0], grid_size[1], grid_spacing[0], d_si, d_sd)
t1 = time.time()
print('backprojection:', t1-t0)
plt.imshow(reco_filtered.buffer)
plt.gray()
plt.show()
