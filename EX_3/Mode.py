import numpy as np
import math
from scipy.fftpack import fft, ifft, fftfreq
from EX_1.phantom import phantom
from EX_1.Grid import Grid
from EX_1.interpolate import interpolate as ip, interpolate
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import time
import pyopencl as cl

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


def create_fanogram(phantom, number_of_projections, detector_spacing, detector_size, angular_increment, d_si, d_sd):
    fanogram = Grid(number_of_projections, detector_size, (angular_increment, detector_spacing))
    fanogram.set_origin((0, -0.5 * (detector_size - 1) * detector_spacing))

    angles = np.deg2rad(np.arange(0, number_of_projections * angular_increment, angular_increment))
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)

    half_detector_size = 0.5 * (detector_size - 1)
    detector_positions = (np.arange(detector_size) - half_detector_size) * detector_spacing

    for beta_idx in range(number_of_projections):
        cos_beta = cos_angles[beta_idx]
        sin_beta = sin_angles[beta_idx]

        # Source position S(-d_si * sin(beta), d_si * cos(beta))
        source_position = np.array([-d_si * sin_beta, d_si * cos_beta])

        for s in range(detector_size):
            s_world = detector_positions[s]

            # Point on the detector
            point_on_detector = source_position + np.array([d_sd * sin_beta + s_world * cos_beta,
                                                            -d_sd * cos_beta + s_world * sin_beta])

            # Ray SP
            ray_SP = point_on_detector - source_position
            SP_length = np.sqrt(ray_SP[0] ** 2 + ray_SP[1] ** 2)
            ray_direction = ray_SP / SP_length

            step_size = 0.5
            number_of_steps = int(np.ceil(SP_length / step_size))

            ray_sum = 0.0
            for i in range(number_of_steps):
                curr_point = source_position + i * step_size * ray_direction
                val = phantom.get_at_physical(curr_point[0], curr_point[1])
                ray_sum += val * step_size

            fanogram.set_at_index(beta_idx, s, ray_sum)

    return fanogram


def backproject_fanbeam(fanogram, size_x, size_y, image_spacing, d_si, d_sd):
    # Create reconstruction grid
    reco = Grid(size_x, size_y, image_spacing)
    reco.set_origin((-0.5 * (size_x - 1) * image_spacing[0], -0.5 * (size_y - 1) * image_spacing[1]))

    angular_increment = fanogram.get_spacing()[0]
    print('angular increment: ' + str(angular_increment))
    # Cosine weighting
    for i in range(fanogram.get_size()[0]):
        for j in range(fanogram.get_size()[1]):
            t = -0.5 * (fanogram.get_size()[1] - 1) * fanogram.get_spacing()[1] + j * fanogram.get_spacing()[1]
            cos_weight = d_sd / np.sqrt(d_sd * 2 + t * 2)
            fanogram.set_at_index(i, j, fanogram.get_at_index(i, j) * cos_weight)

    # Backprojection
    for ix in range(reco.get_size()[0]):
        for iy in range(reco.get_size()[1]):
            x, y = reco.index_to_physical(ix, iy)
            pixel_value = 0.0

            for beta_index in range(fanogram.get_size()[0]):
                beta_degree = beta_index * fanogram.get_spacing()[0] + fanogram.get_origin()[0]
                beta = np.deg2rad(beta_degree)
                source_x = -d_si * np.sin(beta)
                source_y = d_si * np.cos(beta)

                # Calculate coordinates in the fanogram space
                SX = x - source_x
                SY = y - source_y

                t = (d_sd / d_si) * (SX * np.cos(beta) + SY * np.sin(beta))
                #t = -SX * np.cos(beta) - SY * np.sin(beta)
                s = SX * np.sin(beta) - SY * np.cos(beta)
                t_index = (t / fanogram.get_spacing()[1]) + 0.5 * (fanogram.get_size()[1] - 1)

                value = interpolate(fanogram, beta_index, t_index)
                U = (SX * np.sin(beta) - SY * np.cos(beta)) / d_si
                pixel_value += value * U

            reco.set_at_index(ix, iy, pixel_value)

    return reco


def rebinning(fanogram, d_si, d_sd):
    detector_size = fanogram.get_size()[1]
    detector_spacing = fanogram.get_spacing()[1]

    # Create an 180-degree sinogram
    sinogram = Grid(180, detector_size, (1, detector_spacing))
    sinogram.set_origin(fanogram.get_origin())


    for p in range(sinogram.get_size()[0]):
        theta_angle = p * (sinogram.get_spacing()[0]) + (sinogram.get_origin()[0])
        theta = np.deg2rad(theta_angle)  # Convert projection angle to radians

        for s in range(sinogram.get_size()[1]):
            s_world = -0.5 * (detector_size - 1) * detector_spacing + s * detector_spacing
            gamma = np.arctan(s_world / d_si)
            beta = theta - gamma

            # Wrap beta to the range [0, 2 * pi)
            if beta < 0:
                beta_2 = beta + 2 * gamma + np.pi
                gamma_2 = - gamma
                gamma = gamma_2
                beta = beta_2

            s_fan_world = d_sd * np.tan(gamma)
            beta_degrees = np.rad2deg(beta)

            val = fanogram.get_at_physical(beta_degrees, s_fan_world)
            sinogram.set_at_index(p, s, val)

    return sinogram


def backproject_cl(sinogram, size_x, size_y, spacing):
    context = cl.create_some_context()
    queue = cl.CommandQueue(context)

    num_projections, detector_size = sinogram.get_buffer().shape
    angular_increment_degree = np.pi / num_projections
    detector_spacing = spacing[1]
    detector_origin = -0.5 * (detector_size - 1.0) * detector_spacing
    reco_originX = -0.5 * (size_x - 1.0) * spacing[0]
    reco_originY = -0.5 * (size_y - 1.0) * spacing[1]

    sinogram_np = sinogram.get_buffer().astype(np.float32)
    fmt = cl.ImageFormat(cl.channel_order.R, cl.channel_type.FLOAT)
    sinogram_image = cl.Image(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, fmt,
                              shape=(detector_size, num_projections), hostbuf=sinogram_np)
    reco_image = cl.Image(context, cl.mem_flags.WRITE_ONLY, fmt, shape=(size_x, size_y))

    program = cl.Program(context, open(r'C:\Users\RUTVIK\Desktop\CT project\flat-panel-CT-reconstruction\EX_4\backprojection.cl').read()).build()
    kernel = program.backproject

    kernel.set_args(sinogram_image, reco_image, np.int32(num_projections), np.int32(detector_size),
                    np.float32(angular_increment_degree), np.float32(detector_spacing), np.float32(detector_origin),
                    np.int32(size_x), np.int32(size_y), np.float32(reco_originX), np.float32(reco_originY),
                    np.float32(spacing[0]), np.float32(spacing[1]))

    cl.enqueue_nd_range_kernel(queue, kernel, (size_x, size_y), None)

    reco_np = np.empty((size_y, size_x), dtype=np.float32)
    cl.enqueue_copy(queue, reco_np, reco_image, origin=(0, 0), region=(size_x, size_y)).wait()

    reco_grid = Grid(size_y, size_x, spacing)
    reco_grid.set_buffer(reco_np)

    return reco_grid