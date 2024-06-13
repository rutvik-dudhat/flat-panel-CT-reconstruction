from phantom import phantom
import matplotlib.pyplot as plt

shepp_logan = phantom(n = 512, p_type = 'Modified Shepp-Logan', ellipses = None)
plt.imshow(shepp_logan)
plt.gray()
plt.show()


import numpy as np
from interpolate import interpolate  # Assuming interpolate.py contains the interpolation method


class Grid:
    def __init__(self, height, width, spacing):
        self.height = height
        self.width = width
        self.spacing = spacing
        self.origin = (0, 0)
        self.buffer = np.zeros((height, width))

    def set_buffer(self, buffer):
        if buffer.shape == (self.height, self.width):
            self.buffer = buffer
        else:
            raise ValueError("Buffer shape does not match grid dimensions")

    def get_buffer(self):
        return self.buffer

    def get_origin(self):
        return self.origin

    def get_spacing(self):
        return self.spacing

    def get_size(self):
        return self.height, self.width

    def index_to_physical(self, i, j):
        x = self.origin[1] + j * self.spacing
        y = self.origin[0] + i * self.spacing
        return y, x

    def physical_to_index(self, y, x):
        j = int((x - self.origin[1]) / self.spacing)
        i = int((y - self.origin[0]) / self.spacing)
        return i, j

    def set_at_index(self, i, j, value):
        if 0 <= i < self.height and 0 <= j < self.width:
            self.buffer[i, j] = value
        else:
            raise ValueError("Index out of range")

    def get_at_index(self, i, j):
        if 0 <= i < self.height and 0 <= j < self.width:
            return self.buffer[i, j]
        else:
            raise ValueError("Index out of range")

    def get_at_physical(self, y, x):
        i, j = self.physical_to_index(y, x)
        if 0 <= i < self.height - 1 and 0 <= j < self.width - 1:
            # Perform interpolation
            return interpolate(y, x, self.buffer[i:i+2, j:j+2])
        else:
            raise ValueError("Physical coordinates out of range")


# Example usage:
grid = Grid(5, 5, 1)
print("Buffer size:", grid.get_size())
print("Buffer spacing:", grid.get_spacing())
print("Buffer origin:", grid.get_origin())

# Setting values
grid.set_at_index(2, 2, 5)
print("Value at index (2, 2):", grid.get_at_index(2, 2))

# Physical to index conversion
print("Index corresponding to physical coordinates (3.5, 2.5):", grid.physical_to_index(3.5, 2.5))

# Interpolated value at physical coordinates
print("Interpolated value at physical coordinates (3.5, 2.5):", grid.get_at_physical(3.5, 2.5))
