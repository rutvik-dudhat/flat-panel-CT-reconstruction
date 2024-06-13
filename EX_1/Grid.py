import numpy as np
import interpolate as ip
from interpolate import interpolate as ip
#from scipy.interpolate import interpolate
# import your necessary libraries by yourself

# Fill in the missing part of the following functions, class attributes


class Grid:

    def __init__(self, height, width, spacing):
        self.height = height
        self.width = width
        self.spacing = spacing
        self.origin = [-0.5 * (width - 1.0) * spacing[0], -0.5 * (height - 1.0) * spacing[1]]
        # oworld = [ox, oy] = [-0.5(N0-1)s0, -0.5(N1-1)s1]
        # N0 = width, N1 = height, S0 = spacing[0], S1 = spacing[1]
        # to make the origin at the center of the grid
        self.buffer =np.empty((height, width))

    def set_at_index(self, i, j, val):
        self.buffer[i,j] = val
        # set buffer value at the index (i,j)

    def get_at_index(self, i, j):
        return self.buffer[i, j]
        # get buffer value at the index (i,j)

    def get_at_physical(self, x, y):
        i,j=self.physical_to_index(x,y)
        return ip(self,i,j)
        # convert physical to index coordinates and then interpolate


    def index_to_physical(self, i, j):
        x=self.origin[0]+i*self.spacing[0]
        y=self.origin[1]+j*self.spacing[1]
        return x,y
    #convert grid (i,j) to pysical (x,y)


    def physical_to_index(self, x, y):
        return np.array([((x - self.origin[0]) / self.spacing[0]),
                         ((y - self.origin[1]) / self.spacing[1])])
#convert physical (x,y) to grid (i,j)

    def get_size(self):
        return self.height, self.width
#return the size of the grid

    def set_origin(self, origin):
        self.origin = origin


    def get_origin(self):
        return self.origin


    def set_spacing(self, spacing):
        self.spacing = spacing


    def get_spacing(self):
        return self.spacing


    def get_buffer(self):
        return self.buffer


    def set_buffer(self, buffer):
        self.buffer = buffer


