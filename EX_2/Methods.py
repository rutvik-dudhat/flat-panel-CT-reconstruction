import numpy as np
import math
from scipy.fftpack import fft, ifft, fftfreq

from EX_1 import phantom
from EX_1.Grid import Grid
from EX_1.interpolate import interpolate as ip
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from EX_3.Ex3Run import angular_increment
from Ex2Run import detector_spacing


def create_sinogram(phantom, num_projections, detector_spacing, detector_size, scan_range):
    angle_step = scan_range / num_projections
    sinogram = Grid(num_projections, detector_size, (angle_step, detector_spacing))
    sinogram.set_origin((0, -0.5 * (detector_size - 1) * detector_spacing))

    for proj_index in range(num_projections):
        theta = proj_index * angle_step
        cos_theta = np.cos(np.deg2rad(theta))
        sin_theta = np.sin(np.deg2rad(theta))

        for det_index in range(detector_size):
            det_pos = -0.5 * detector_size * detector_spacing + det_index * detector_spacing
            line_integral = 0.0
            sample_step = detector_spacing

            for t in np.arange(
                -0.5 * np.sqrt(phantom.get_size()[0]**2 + phantom.get_size()[1]**2) * phantom.get_spacing()[0],
                0.5 * np.sqrt(phantom.get_size()[0]**2 + phantom.get_size()[1]**2) * phantom.get_spacing()[1],
                sample_step
            ):
                x = t * cos_theta - det_pos * sin_theta
                y = t * sin_theta + det_pos * cos_theta
                line_integral += phantom.get_at_physical(x, y) * sample_step

            sinogram.set_at_index(proj_index, det_index, line_integral)

    return sinogram

def backproject(sinogram, grid_width, grid_height, grid_spacing):
    reconstruction = Grid(grid_width, grid_height, grid_spacing)
    reconstruction.set_origin((-0.5 * (grid_width - 1) * grid_spacing[0], -0.5 * (grid_height - 1) * grid_spacing[1]))

    for grid_x in range(grid_width):
        for grid_y in range(grid_height):
            x, y = reconstruction.index_to_physical(grid_x, grid_y)
            pixel_sum = 0.0

            for angle_index in range(sinogram.get_size()[0]):
                theta = angle_index * sinogram.get_spacing()[0]
                cos_theta = np.cos(np.deg2rad(theta))
                sin_theta = np.sin(np.deg2rad(theta))
                s = x * sin_theta + y * cos_theta
                s_index = (s - sinogram.get_origin()[1]) / sinogram.get_spacing()[1]
                pixel_sum += ip(sinogram, angle_index, s_index)

            reconstruction.set_at_index(grid_x, grid_y, pixel_sum / sinogram.get_size()[0])

    return reconstruction


def ramp_filter(sinogram, detector_spacing):
    num_projections, num_detectors = sinogram.get_size()
    filter_len = next_power_of_two(num_detectors)
    filter_freq = fftfreq(filter_len, detector_spacing)
    ramp = np.abs(filter_freq)
    #plt.plot(ramp)
    #plt.title('Ramp Filter')
    #plt.show()
    sinogram_padded = np.pad(sinogram.buffer, ((0, 0), (0, filter_len - num_detectors)), mode='constant')
    sinogram_fft = fft(sinogram_padded, axis=1)
    filtered_fft = sinogram_fft * ramp
    filtered_sinogram = ifft(filtered_fft, axis=1).real[:, :num_detectors]

    filtered_sinogram_grid = Grid(num_projections, num_detectors, sinogram.get_spacing())
    filtered_sinogram_grid.set_buffer(filtered_sinogram)
    filtered_sinogram_grid.set_origin(sinogram.get_origin())

    return filtered_sinogram_grid


def ramlak_filter(sinogram, detector_spacing):
    sinogram_buffer = sinogram.get_buffer()
    num_projections, detector_size = sinogram.get_size()
    n = detector_size // 2

    # Create the Ram-Lak filter kernel
    kernel_size = detector_size+1
    kernel = np.zeros(kernel_size)
    j_vals = np.arange(-n, n + 1)
    kernel[n] = 0.25 / (detector_spacing ** 2)
    kernel[1::2] = -1 / (np.pi ** 2 * j_vals[1::2] ** 2 * detector_spacing ** 2)
    #plt.plot(kernel)
    #plt.title('Ram-Lak Filter')
    #plt.show()
    # Apply the filter to each projection using convolution
    filtered_sinogram = np.zeros_like(sinogram_buffer)
    for i in range(sinogram.get_size()[0]):
        projection = sinogram_buffer[i, :]
        filtered_projection = convolve(projection, kernel, mode='reflect')
        filtered_sinogram[i, :] = filtered_projection

    sinogram.set_buffer(filtered_sinogram)

    return sinogram

def next_power_of_two(value):
    return 1 << (value - 1).bit_length()



