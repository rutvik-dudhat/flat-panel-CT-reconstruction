import time
import OpenCLMethods  # Assuming this is correctly updated with the OpenCL methods
from EX_1.Grid import Grid
import numpy as np

height = 10000
width = 5000
np_grid1 = np.zeros((height, width))
np_grid2 = np.zeros((height, width))

# initialization
t0 = time.time()
for i in range(height):
    for j in range(width):
        np_grid1[i, j] = i*j
        np_grid2[i, j] = i*j
grid1 = Grid(height, width, [1, 1])
grid1.set_buffer(np_grid1)

grid2 = Grid(height, width, [1, 1])
grid2.set_buffer(np_grid2)

grid3 = Grid(height, width, [1, 1])
t1 = time.time()
print("Time for grid initialization: ", t1-t0)

# Test addition with for loop
t0 = time.time()
for i in range(height):
    for j in range(width):
        grid3.set_at_index(i, j, grid1.get_at_index(i, j) + grid2.get_at_index(i, j))
t1 = time.time()
print("Time for for-loop addition: ", t1-t0)
print("grid3[{}, {}]: ".format(height - 1, width - 1), grid3.get_at_index(height - 1, width - 1))

# Test addition with numpy
np_grid3 = np.add(np_grid1, np_grid2)
t1 = time.time()
print("Time for numpy addition: ", t1-t0)
print("np_grid3[{}, {}]: ".format(height - 1, width - 1), np_grid3[height - 1, width - 1])

# Test addition with OpenCL texture
t0 = time.time()
grid3 = OpenCLMethods.addGrids_texture(grid1, grid2)
t1 = time.time()
print("Time for OpenCL texture addition: ", t1-t0)
print("grid3[{}, {}]: ".format(height - 1, width - 1), grid3.get_at_index(height - 1, width - 1))

t0 = time.time()
grid4 = OpenCLMethods.addGrids_normal(grid1, grid2)
t1 = time.time()
print("Time for OpenCL normal buffer addition: ", t1-t0)
print("grid3[{}, {}]: ".format(height - 1, width - 1), grid3.get_at_index(height - 1, width - 1))


# For OpenCL back projection, please see backprojection.cl file and Methods.backproject_cl(sinogram, reco_sizeX, reco_sizeY, reco_spacing)

print('For OpenCL back projection, please see backprojection.cl file and Methods.backproject_cl(sinogram, reco_sizeX, reco_sizeY, reco_spacing)')







