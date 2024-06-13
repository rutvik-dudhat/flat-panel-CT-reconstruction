import numpy as np
import math
from scipy.fftpack import fft, ifft, fftfreq
from EX_1.phantom import phantom
from EX_1.Grid import Grid
from EX_1.interpolate import interpolate as ip
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import time

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

    sinogram_padded = np.pad(sinogram.buffer, ((0, 0), (0, filter_len - num_detectors)), mode='constant')
    sinogram_fft = fft(sinogram_padded, axis=1)
    filtered_fft = sinogram_fft * ramp
    filtered_sinogram = ifft(filtered_fft, axis=1).real[:, :num_detectors]

    filtered_sinogram_grid = Grid(num_projections, num_detectors, sinogram.get_spacing())
    filtered_sinogram_grid.set_buffer(filtered_sinogram)
    filtered_sinogram_grid.set_origin(sinogram.get_origin())

    return filtered_sinogram_grid

def next_power_of_two(value):
    return 1 << (value - 1).bit_length()

def create_fanogram(phantom, number_of_projections, detector_spacing, detector_sizeInPixels, angular_increment, d_si, d_sd):
    fanogram = Grid(number_of_projections, detector_sizeInPixels, (angular_increment, detector_spacing))
    fanogram.set_origin((0, -0.5 * (detector_sizeInPixels - 1) * detector_spacing))

    angles = np.arange(0, number_of_projections * angular_increment, angular_increment)

    for proj_index, beta in enumerate(angles):
        beta_rad = np.deg2rad(beta)
        cos_beta = np.cos(beta_rad)
        sin_beta = np.sin(beta_rad)

        for det_index in range(detector_sizeInPixels):
            t = (det_index - 0.5 * (detector_sizeInPixels - 1)) * detector_spacing
            line_integral = 0.0
            sample_step = detector_spacing

            for s in np.arange(-phantom.get_size()[0] / 2, phantom.get_size()[0] / 2, sample_step):
                x = s * cos_beta - t * sin_beta + d_si * cos_beta
                y = s * sin_beta + t * cos_beta + d_si * sin_beta
                line_integral += phantom.get_at_physical(x, y) * sample_step

            fanogram.set_at_index(proj_index, det_index, line_integral)

    return fanogram

def rebinning(fanogram, d_si, d_sd):
    num_projections, num_detectors = fanogram.get_size()
    detector_spacing = fanogram.get_spacing()[1]
    angular_increment = fanogram.get_spacing()[0]

    max_t = (num_detectors - 1) * detector_spacing / 2
    s_values = np.linspace(-max_t, max_t, num_detectors)
    theta_values = np.linspace(0, num_projections * angular_increment, num_projections)

    sinogram = Grid(num_projections, num_detectors, (angular_increment, detector_spacing))
    sinogram.set_origin((0, -0.5 * (num_detectors - 1) * detector_spacing))

    for proj_index in range(num_projections):
        beta = np.deg2rad(proj_index * angular_increment)

        for det_index in range(num_detectors):
            s = s_values[det_index]
            t = s * (d_sd / d_si)
            t_index = (t - fanogram.get_origin()[1]) / detector_spacing

            if 0 <= t_index < num_detectors:
                value = ip(fanogram, proj_index, t_index)
            else:
                t = -s * (d_sd / d_si)
                t_index = (t - fanogram.get_origin()[1]) / detector_spacing
                value = ip(fanogram, proj_index, t_index) if 0 <= t_index < num_detectors else 0

            sinogram.set_at_index(proj_index, det_index, value)

    return sinogram

def backproject_fanbeam(sinogram, grid_width, grid_height, grid_spacing, d_si, d_sd):
    if isinstance(grid_spacing, (int, float)):
        grid_spacing = (grid_spacing, grid_spacing)

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
