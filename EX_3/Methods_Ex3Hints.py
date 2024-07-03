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

    sinogram = Grid(num_projections, num_detectors, (angular_increment, detector_spacing))
    sinogram.set_origin((0, -0.5 * (num_detectors - 1) * detector_spacing))

    max_t = (num_detectors - 1) * detector_spacing / 2
    s_values = np.linspace(-max_t, max_t, num_detectors)

    for p in range(0, num_projections):
        theta_degree = p * angular_increment
        theta = np.deg2rad(theta_degree)

        for s in range(0, num_detectors):
            s_world = s_values[s]
            gamma = np.arctan(s_world / d_sd)
            beta = theta + gamma

            if beta < 0:
                beta += 2 * np.pi

            s_fan_world = d_si * np.tan(gamma)
            beta_degrees = np.rad2deg(beta)
            val = ip(fanogram, int(beta_degrees / angular_increment), s_fan_world / detector_spacing)

            sinogram.set_at_index(p, s, val)

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


###################################################

def create_fanogram(phantom, number_of_projections, detector_spacing, detector_size, angular_increment, d_si, d_sd):
    fanogram = Grid(number_of_projections, detector_size, (angular_increment, detector_spacing))
    fanogram.set_origin((0, -0.5 * (detector_size - 1) * detector_spacing))

    for beta_idx in range(0, number_of_projections):
        beta = np.deg2rad(beta_idx * angular_increment)

        source_position = np.array([-d_si * np.sin(beta), d_si * np.cos(beta)])
        source_direction = np.array([np.cos(beta), np.sin(beta)])
        detector_direction = np.array([-np.sin(beta), np.cos(beta)])

        for s in range(0, detector_size):
            s_world = (s - 0.5 * (detector_size - 1)) * detector_spacing
            point_on_detector = source_position + d_sd * detector_direction + s_world * source_direction
            ray_SP = point_on_detector - source_position
            SP_length = np.linalg.norm(ray_SP)
            ray_direction = ray_SP / SP_length

            step_size = 0.5
            number_of_steps = math.ceil(SP_length / step_size)

            ray_sum = 0
            for i in range(0, number_of_steps):
                curr_point = source_position + i * step_size * ray_direction
                val = phantom.get_at_physical(curr_point[0], curr_point[1])
                ray_sum += val * step_size

            fanogram.set_at_index(beta_idx, s, ray_sum)

    return fanogram

def rebinning(fanogram, d_si, d_sd):
    num_projections, num_detectors = fanogram.get_size()
    detector_spacing = fanogram.get_spacing()[1]
    angular_increment = fanogram.get_spacing()[0]

    sinogram = Grid(num_projections, num_detectors, (angular_increment, detector_spacing))
    sinogram.set_origin((0, -0.5 * (num_detectors - 1) * detector_spacing))

    max_t = (num_detectors - 1) * detector_spacing / 2
    s_values = np.linspace(-max_t, max_t, num_detectors)

    for p in range(0, num_projections):
        theta_degree = p * angular_increment
        theta = np.deg2rad(theta_degree)

        for s in range(0, num_detectors):
            s_world = s_values[s]
            gamma = np.arctan(s_world / d_sd)
            beta = theta + gamma

            if beta < 0:
                beta += 2 * np.pi

            s_fan_world = d_si * np.tan(gamma)
            beta_degrees = np.rad2deg(beta)
            val = ip(fanogram, int(beta_degrees / angular_increment), s_fan_world / detector_spacing)

            sinogram.set_at_index(p, s, val)

    return sinogram

def backproject_fanbeam(fanogram, size_x, size_y, image_spacing, d_si, d_sd):
    reco = Grid(size_x, size_y, image_spacing)
    reco.set_origin((-0.5 * (size_x - 1) * image_spacing[0], -0.5 * (size_y - 1) * image_spacing[1]))

    d_id = d_sd - d_si
    angular_range = 360
    angular_increment = fanogram.get_spacing()[0]
    print('angular increment: ' + str(angular_increment))

    # Cosine weighting
    for i in range(0, fanogram.get_size()[0]):
        for j in range(0, fanogram.get_size()[1]):
            beta = i * angular_increment
            t = j * fanogram.get_spacing()[1]
            cos_weight = d_si / (d_si + t)
            fanogram.set_at_index(i, j, fanogram.get_at_index(i, j) * cos_weight)

    for i_x in range(0, reco.get_size()[0]):
        for i_y in range(0, reco.get_size()[1]):
            x, y = reco.index_to_physical(i_x, i_y)

            for beta_index in range(0, fanogram.get_size()[0]):
                beta_degree = beta_index * angular_increment
                beta = np.deg2rad(beta_degree)
                source = np.array([d_si * np.cos(beta), d_si * np.sin(beta)])
                SX = np.array([x - source[0], y - source[1]])
                SQ = np.array([x, y])
                ratio_alpha = np.linalg.norm(SQ) / np.linalg.norm(SX)
                SP = ratio_alpha * SX

                t = SP[0] * np.sin(beta) + SP[1] * np.cos(beta)
                s_index = (t - fanogram.get_origin()[1]) / fanogram.get_spacing()[1]
                value = ip(fanogram, beta_index, s_index)
                U = (d_id / (d_id + np.linalg.norm(SX)))
                value *= U
                reco.set_at_index(i_x, i_y, reco.get_at_index(i_x, i_y) + value)

    return reco
