
import numpy
import Grid as grid
from skimage.transform import iradon
import numpy as np
import pyopencl as cl
import matplotlib.pyplot as plt
import Helpers.Utility_functions as helper
from scipy.fftpack import fft, ifft, fftfreq


def backprojectOpenCL(sinogram, recon_size_x, recon_size_y, spacing):
    recon_img = grid.Grid(recon_size_x, recon_size_y, spacing)  # spacing should be received as a tuple
    deltaS = sinogram.get_spacing()[1]
    deltaTheta = sinogram.get_spacing()[0]
    detector_length, projections = sinogram.get_size()[1], sinogram.get_size()[0]
    sinogram.set_origin(0, -(detector_length - 1) * deltaS / 2)
    sino_origin = sinogram.get_origin()
    spacing = recon_img.get_spacing()
    origin = recon_img.get_origin()

    # setting up device and queuing kernel call
    platform = cl.get_platforms()
    GPU = platform[0].get_devices()
    ctx = cl.Context(GPU)
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags

    # build a 2D OpenCL Image from the numpy array, buffers and output image
    src = numpy.array(sinogram.get_buffer(), dtype=numpy.float32)
    sino_buffer = cl.image_from_array(ctx, src, num_channels=1, mode="r", norm_int=False)

    # build destination image texture
    fmt = cl.ImageFormat(cl.channel_order.INTENSITY, cl.channel_type.UNSIGNED_INT8)
    dest_buf = cl.Image(ctx, mf.WRITE_ONLY, fmt, shape=(recon_size_x, recon_size_y))
    res_np = numpy.empty_like(recon_img.buffer)

    # Passing scalar params from host to device.
    kernel = open("backproject.cl").read()
    prg = cl.Program(ctx, kernel).build()
    prg.backproject(queue, [recon_size_x, recon_size_y], None, sino_buffer, dest_buf,
                    np.float32(deltaS), np.float32(deltaTheta), np.float32(sino_origin[0]), np.float32(sino_origin[1]),
                    np.float32(detector_length), np.float32(projections), np.float32(spacing[0]),
                    np.float32(spacing[1]),
                    np.float32(origin[0]), np.float32(spacing[1]))

    cl.enqueue_copy(queue, res_np, dest_buf, origin=(0, 0), region=(recon_size_x, recon_size_y))

    correction_factor = np.pi / (2 * projections)
    img_recon_corrected = res_np * correction_factor
    return img_recon_corrected


def create_sinogram(phantom, projections, detector_spacing, detector_size, angular_scan_range):
    grid_phantom = grid.Grid(len(phantom), len(phantom[0]), (1, 1))
    grid_phantom.set_buffer(phantom)
    angular_step = angular_scan_range / projections
    detector_length = detector_spacing * detector_size

    theta = np.linspace(0, angular_scan_range, projections)
    num_of_Angles = len(theta)
    s = np.linspace(-(detector_length - 1) * detector_spacing / 2, (detector_length - 1) * detector_spacing / 2,
                    detector_size)
    grid_sino = grid.Grid(detector_size, num_of_Angles, (angular_step, detector_spacing))
    grid_sino.set_origin(0, -(detector_length - 1) * detector_spacing / 2)
    for i in range(detector_size):
        for j in range(num_of_Angles):
            grid_sino.set_at_index(i, j, helper.line_integral(grid_phantom, s[i], theta[j]))
    return grid_sino


def ramp_filter(sinogram, detector_spacing):
    # Zero padding calculation and calculating ramp filter kernel
    projection_size_padded = helper.next_power_of_two(sinogram.buffer.shape[0])
    pad_width = np.zeros([(projection_size_padded - sinogram.buffer.shape[0]) // 2, sinogram.buffer.shape[1]])
    padded_sino = np.vstack((pad_width, sinogram.buffer))
    padded_sino = np.vstack((padded_sino, pad_width))
    delta_f = 1 / (detector_spacing * projection_size_padded)
    f = fftfreq(projection_size_padded, d=delta_f).reshape(-1, 1)
    fourier_filter = 2 * np.abs(f)

    # Optional : plotting the filter kernel in freq domain

    # Apply ramp filter in Fourier domain. Filtering row wise (along each detector line)
    sino_filtered = grid.Grid(padded_sino.shape[0], padded_sino.shape[1],
                              (sinogram.get_spacing()[0], sinogram.get_spacing()[1]))
    sino_fft = fft(padded_sino, axis=0)
    freq_filtered = sino_fft * fourier_filter
    sino_filtered.buffer = np.real(ifft(freq_filtered, axis=0))

    # Resizing to original sinogram dimensions
    sino_filtered.buffer = sino_filtered.buffer[pad_width.shape[0]:pad_width.shape[0] + sinogram.buffer.shape[0], :]
    sino_filtered.height, sino_filtered.width = sino_filtered.get_buffer().shape[0], sino_filtered.get_buffer().shape[1]

    return sino_filtered


def ramlak_filter(sinogram, detector_spacing):
    projection_size_padded = helper.next_power_of_two(sinogram.buffer.shape[0])
    pad_width = np.zeros([(projection_size_padded - sinogram.buffer.shape[0]) // 2, sinogram.buffer.shape[1]])
    padded_sino = np.vstack((pad_width, sinogram.buffer))
    padded_sino = np.vstack((padded_sino, pad_width))

    constantFactor = -1.0 / (float(np.power(np.pi, 2) * np.power(detector_spacing, 2)))
    n1 = np.arange(1, int(projection_size_padded / 2) + 1, 2)
    n2 = np.arange(int(projection_size_padded / 2) - 1, 0, -2)
    n = np.concatenate((n1, n2))
    filter_array = np.zeros(projection_size_padded)
    filter_array[0] = 1 / (4 * np.power(detector_spacing, 2))
    filter_array[1::2] = constantFactor / np.power(n, 2)

    # Ram Lak convolver initialized in spatial domain. Computing it's FT with sinogram instead
    # of convolving in spatial domain
    fourier_filter = 2 * np.real(fft(filter_array)).reshape(-1, 1)

    # Optional : plotting the filter kernel in spatial domain
    plt.plot(filter_array)
    plt.show()
    # Optional : plotting the filter kernel in freq domain
    plt.plot(fourier_filter)
    plt.show()

    # Apply ramp filter in Fourier domain. Filtering row wise (along each detector line)
    sino_filtered = grid.Grid(padded_sino.shape[0], padded_sino.shape[1],
                              (sinogram.get_spacing()[0], sinogram.get_spacing()[1]))
    sino_fft = fft(padded_sino, axis=0)
    freq_filtered = sino_fft * fourier_filter
    sino_filtered.buffer = np.real(ifft(freq_filtered, axis=0))

    # Resizing to original sinogram dimensions
    sino_filtered.buffer = sino_filtered.buffer[pad_width.shape[0]:pad_width.shape[0] + sinogram.buffer.shape[0], :]
    sino_filtered.height, sino_filtered.width = sino_filtered.get_buffer().shape[0], sino_filtered.get_buffer().shape[1]

    return sino_filtered


def backproject(sinogram, recon_size_x, recon_size_y, spacing):
    recon_img = grid.Grid(recon_size_x, recon_size_y, spacing)  # spacing should be received as a tuple
    detector_length = sinogram.get_size()[1]
    deltaS = sinogram.get_spacing()[1]
    projections = len(sinogram.get_buffer()[0])

    theta = np.linspace(0, 180, projections, endpoint=False)

    # Using library for to create FBP for comparison
    img_fbp = iradon(sinogram.get_buffer(), theta=theta, output_size=recon_size_x, filter_name=None, circle=False)

    for x in range(recon_img.get_size()[0]):
        for y in range(recon_img.get_size()[1]):
            w = recon_img.index_to_physical(x, y)
            for i in range(0, len(theta)):  # last loop
                angle = (theta[i]) * (np.pi / 180)
                s = w[0] * (recon_img.get_spacing()[0] * np.cos(angle)) + w[1] * (
                        recon_img.get_spacing()[0] * np.sin(angle))

                # compute detector element index from world coordinates
                s += detector_length / 2
                s /= deltaS

                if sinogram.get_size()[0] >= s + 1 and s > 0:
                    val = sinogram.get_at_index(int(np.floor(s)), i)
                    recon_img.buffer[x][y] += val

    correction_factor = np.pi / (2 * len(theta))
    img_recon_corrected = recon_img.get_buffer() * correction_factor
    return img_recon_corrected, img_fbp
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
