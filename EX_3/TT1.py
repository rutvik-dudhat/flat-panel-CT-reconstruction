import numpy as np
import math
from scipy.fftpack import fft, ifft, fftfreq
from EX_1.Grid import Grid
from EX_1.interpolate import interpolate as ip
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from Ex2Run import detector_spacing
from EX_2.Methods import create_sinogram, backproject, ramp_filter, ramlak_filter

# Fan-Beam Projection
def create_fanogram(phantom, number_of_projections, detector_spacing, detector_size, angular_increment, d_si, d_sd):
    fanogram = Grid(number_of_projections, detector_size)
    fanogram.set_origin((0, 0))

    for beta_idx in range(number_of_projections):
        beta = beta_idx * angular_increment
        source_position = np.array([d_si * np.cos(beta), d_si * np.sin(beta)])
        for det_index in range(detector_size):
            det_pos = (det_index - detector_size / 2) * detector_spacing
            point_on_detector = source_position + d_sd * np.array([-np.sin(beta), np.cos(beta)]) + det_pos * np.array([np.cos(beta), np.sin(beta)])
            ray = point_on_detector - source_position
            ray_length = np.linalg.norm(ray)
            ray_dir = ray / ray_length

            step_size = detector_spacing
            steps = np.arange(0, ray_length, step_size)
            line_integral = 0

            for t in steps:
                sample_point = source_position + t * ray_dir
                line_integral += phantom.get_at_physical(sample_point[0], sample_point[1]) * step_size

            fanogram.set_at_index(beta_idx, det_index, line_integral)

    return fanogram

# Rebinning
def rebinning(fanogram, d_si, d_sd):
    num_projections, detector_size = fanogram.get_size()
    delta_beta = 2 * np.pi / num_projections
    delta_theta = np.arcsin(detector_spacing / (2 * d_si))

    num_theta = int(np.ceil(np.pi / delta_theta))
    sinogram = Grid(num_theta, detector_size)
    sinogram.set_origin((0, 0))

    for p in range(num_theta):
        theta = p * delta_theta
        for s in range(detector_size):
            t = (s - detector_size / 2) * detector_spacing
            gamma = np.arctan(t / d_si)
            beta = theta + gamma
            if beta >= 2 * np.pi:
                beta -= 2 * np.pi
            if beta < 0:
                beta += 2 * np.pi

            beta_idx = int(beta / delta_beta)
            sinogram.set_at_index(p, s, fanogram.get_at_index(beta_idx, s))

    return sinogram

# Fan-Beam Backprojection
def backproject_fanbeam(fanogram, grid_width, grid_height, grid_spacing, d_si, d_sd):
    reconstruction = Grid(grid_width, grid_height, grid_spacing)
    reconstruction.set_origin((-0.5 * (grid_width - 1) * grid_spacing[0], -0.5 * (grid_height - 1) * grid_spacing[1]))

    num_projections, detector_size = fanogram.get_size()
    delta_beta = 2 * np.pi / num_projections

    for grid_x in range(grid_width):
        for grid_y in range(grid_height):
            x, y = reconstruction.index_to_physical(grid_x, grid_y)
            pixel_value = 0

            for beta_idx in range(num_projections):
                beta = beta_idx * delta_beta
                source_position = np.array([d_si * np.cos(beta), d_si * np.sin(beta)])
                diff = np.array([x, y]) - source_position
                distance = np.linalg.norm(diff)
                alpha = np.arctan2(diff[1], diff[0]) - beta
                t = d_si * np.tan(alpha)
                det_index = int((t / detector_spacing) + detector_size / 2)

                if 0 <= det_index < detector_size:
                    pixel_value += fanogram.get_at_index(beta_idx, det_index) / (distance * distance)

            reconstruction.set_at_index(grid_x, grid_y, pixel_value * delta_beta)

    return reconstruction
