import pyopencl as cl
import numpy as np
from EX_1.Grid import Grid


def addGrids_normal(grid1, grid2):
    context = cl.create_some_context()
    queue = cl.CommandQueue(context)

    height, width = grid1.get_size()
    grid1_np = grid1.get_buffer().astype(np.float32)
    grid2_np = grid2.get_buffer().astype(np.float32)

    mf = cl.mem_flags
    grid1_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=grid1_np)
    grid2_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=grid2_np)
    result_buf = cl.Buffer(context, mf.WRITE_ONLY, grid1_np.nbytes)

    program = cl.Program(context, open('grid_addition.cl').read()).build()
    kernel = program.add_grids

    kernel.set_args(grid1_buf, grid2_buf, result_buf, np.int32(width), np.int32(height))
    cl.enqueue_nd_range_kernel(queue, kernel, (width, height), None)

    result_np = np.empty_like(grid1_np)
    cl.enqueue_copy(queue, result_np, result_buf).wait()

    result_grid = Grid(height, width, grid1.get_spacing())
    result_grid.set_buffer(result_np)

    return result_grid


def addGrids_texture(grid1, grid2):
    context = cl.create_some_context()
    queue = cl.CommandQueue(context)

    height, width = grid1.get_size()
    grid1_np = grid1.get_buffer().astype(np.float32)
    grid2_np = grid2.get_buffer().astype(np.float32)

    fmt = cl.ImageFormat(cl.channel_order.R, cl.channel_type.FLOAT)
    grid1_image = cl.Image(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, fmt, shape=(width, height),
                           hostbuf=grid1_np)
    grid2_image = cl.Image(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, fmt, shape=(width, height),
                           hostbuf=grid2_np)
    result_image = cl.Image(context, cl.mem_flags.WRITE_ONLY, fmt, shape=(width, height))

    program = cl.Program(context, open('grid_addition.cl').read()).build()
    kernel = program.add_grids_image

    kernel.set_args(grid1_image, grid2_image, result_image)
    cl.enqueue_nd_range_kernel(queue, kernel, (width, height), None)

    result_np = np.empty((height, width), dtype=np.float32)
    cl.enqueue_copy(queue, result_np, result_image, origin=(0, 0), region=(width, height)).wait()

    result_grid = Grid(height, width, grid1.get_spacing())
    result_grid.set_buffer(result_np)

    return result_grid





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

    program = cl.Program(context, open('backprojection.cl').read()).build()
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
