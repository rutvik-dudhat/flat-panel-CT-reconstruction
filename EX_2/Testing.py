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
                -0.5 * np.sqrt(phantom.get_size()[0]*2 + phantom.get_size()[1]*2) * phantom.get_spacing()[0],
                0.5 * np.sqrt(2 * phantom.get_size()[0]*2 + phantom.get_size()[1]*2) * phantom.get_spacing()[1],
                0.5
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
                s_index = (-sinogram.get_origin()[1] + s) / sinogram.get_spacing()[1]
                pixel_sum += interpolate(sinogram, angle_index, s_index)

            reconstruction.set_at_index(grid_x, grid_y, pixel_sum)

    return reconstruction


    ########################################

    def backproject(sinogram, size_x, size_y, grid_spacing):
        reco = Grid(size_x, size_y, grid_spacing)
        # reco = Grid(size_x, size_y, 1, 0, grid_spacing, 0, grid_spacing)
        # reco.origin = (-0.5 * (number_of_projection - 1)**2 * grid_spacing, 0)
        reco.origin = (-0.5 * size_x * grid_spacing[0], -0.5 * size_y * grid_spacing[1])

        for i in range(size_x):
            for j in range(size_y):
                for k in range(sinogram.width):
                    angle = k * 2 * math.pi / sinogram.width
                    s = math.sqrt((i - sinogram.origin[0]) ** 2 + (j - sinogram.origin[1]) ** 2)
                    x = s * math.cos(angle) + sinogram.origin[0]
                    y = s * math.sin(angle) + sinogram.origin[1]
                    if 0 <= x < sinogram.width and 0 <= y < sinogram.height:
                        reco.set_at_index(i, j, reco.get_at_index(i, j) + sinogram.get_at_index(int(x), k))

        return reco

    def ramp_filter(sinogram, detector_spacing):
        filter_array = np.zeros(sinogram.width)
        for i in range(sinogram.width):
            f = i / sinogram.width
            filter_array[i] = f if f > 0 and f < 0.5 else 0
            sinogram_filtered = Grid(sinogram.height, sinogram.width, (1, detector_spacing))
            sinogram_filtered.set_buffer(filter_array)
            plt.imshow(sinogram_filtered.buffer.reshape(-1, 1))
            return sinogram_filtered
        return Grid(sinogram.height, sinogram.width, (1, detector_spacing)).set_buffer(filter_array)

    def ramlak_filter(sinogram, detector_spacing):
        filter_array = np.zeros(sinogram.width)
        for i in range(sinogram.width):
            if i == sinogram.width // 2:
                filter_array[i] = 1
            elif i < sinogram.width // 2:
                filter_array[i] = math.sin(math.pi * i / sinogram.width) / (math.pi * i / sinogram.width)
            else:
                filter_array[i] = -math.sin(math.pi * (i - sinogram.width) / sinogram.width) / (
                            math.pi * (i - sinogram.width) / sinogram.width)
        return Grid(filter_array, 1, sinogram.width, 0, 1, 0, 1)

    def next_power_of_two(value):
        if is_power_of_two(value):
            return value * 2
        else:
            i = 2
            while i <= value:
                i *= 2
            return i * 2

    def is_power_of_two(k):
        return k and not k & (k - 1)

#########################################################################################

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

    return filtered_sinogram_grid

def ramlak_filter(sinogram, detector_spacing):
    num_projections, num_detectors = sinogram.get_size()
    filter_array = np.zeros(num_detectors)
    center = num_detectors // 2

    for i in range(num_detectors):
        if i == center:
            filter_array[i] = 1 / (4 * detector_spacing ** 2)
        elif (i - center) % 2 == 1:
            filter_array[i] = -1 / (math.pi ** 2 * detector_spacing ** 2 * (i - center) ** 2)

    sinogram_padded = np.pad(sinogram.buffer, ((0, 0), (center, center)), mode='constant')
    filtered_sinogram = np.zeros_like(sinogram.buffer)

    for i in range(num_projections):
        filtered_sinogram[i] = np.convolve(sinogram_padded[i], filter_array, mode='same')[center:-center]

    filtered_sinogram_grid = Grid(num_projections, num_detectors, sinogram.get_spacing())
    filtered_sinogram_grid.set_buffer(filtered_sinogram)

    return filtered_sinogram_grid

def next_power_of_two(value):
    return 1 << (value - 1).bit_length()

def is_power_of_two(k):
    return k and not k & (k - 1)
######################################################################

def ramlak_filter(sinogram, detector_spacing):
    num_projections, num_detectors = sinogram.get_size()
    filter_len = next_power_of_two(num_detectors)

    # Create the Ram-Lak filter in the time domain
    ramlak = np.zeros(filter_len)
    for i in range(filter_len):
        if i == filter_len // 2:
            ramlak[i] = 1
        elif i % 2 == 0:
            ramlak[i] = 0
        else:
            ramlak[i] = -4 / (math.pi ** 2 * (i - filter_len // 2) ** 2)

    # FFT of the filter
    ramlak = fft(ramlak)

    # Pad the sinogram
    sinogram_padded = np.pad(sinogram.buffer, ((0, 0), (0, filter_len - num_detectors)), mode='constant')

    # FFT of the padded sinogram
    sinogram_fft = fft(sinogram_padded, axis=1)

    # Apply the filter in the frequency domain
    filtered_fft = sinogram_fft * ramlak

    # Inverse FFT to get the filtered sinogram
    filtered_sinogram = ifft(filtered_fft, axis=1).real[:, :num_detectors]

    # Set the filtered sinogram back into a Grid object
    filtered_sinogram_grid = Grid(num_projections, num_detectors, sinogram.get_spacing())
    filtered_sinogram_grid.set_buffer(filtered_sinogram)
    filtered_sinogram_grid.set_origin(sinogram.get_origin())

    return filtered_sinogram_grid
###################################################

##Mohit no code

def ramlak_filter(sinogram, detector_spacing):
    sinogram_buffer = sinogram.get_buffer()
    num_projections, detector_size = sinogram.get_size()
    n = detector_size // 2

    # Create the Ramachandran-Lakshminarayanan filter kernel
    kernel_size = detector_size + 1
    kernel = np.zeros(kernel_size)
    j_vals = np.arange(-n, n + 1)
    kernel[n] = 0.25 / (detector_spacing ** 2)
    kernel[1::2] = -1 / (np.pi ** 2 * j_vals[1::2] ** 2 * detector_spacing ** 2)

    # Apply the filter to each projection using convolution
    filtered_sinogram = np.zeros_like(sinogram_buffer)
    for i in range(sinogram.get_size()[0]):
        projection = sinogram_buffer[i, :]
        filtered_projection = convolve(projection, kernel, mode='reflect')
        filtered_sinogram[i, :] = filtered_projection

    sinogram.set_buffer(filtered_sinogram)

    return sinogram
