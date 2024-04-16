import numpy as np


# import your necessary libraries by yourself

# Fill in the missing part of the following functions, class attributes

class Grid:

    def __init__(self, height, width, spacing):
        self.height = height
        self.width = width
        self.spacing = spacing
        self.origin = (0, 0)
        self.buffer =np.zeros((height, width))

    def set_at_index(self, i, j, val):
        self.buffer[i,j] = val


    def get_at_index(self, i, j):
        return self.buffer[i,j]


    def get_at_physical(self, x, y):
        i,j =self.physical_to_index(x,y)
        return self.get_at_index(i,j)


    def index_to_physical(self, i, j):
        #return i*self.spacing, j*self.spacing
        x=self.origin[0]+i*self.spacing
        y=self.origin[1]+j*self.spacing
        return x,y


    def physical_to_index(self, x, y):
        i=int((x-self.origin[0])/self.spacing)
        j=int((y-self.origin[1])/self.spacing)
        return i,j


    def get_size(self):
        return self.height, self.width


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


