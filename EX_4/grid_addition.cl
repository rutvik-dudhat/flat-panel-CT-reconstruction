__kernel void add_grids(__global const float* grid1, __global const float* grid2, __global float* result, int width, int height) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i < width && j < height) {
        int index = i * height + j;
        result[index] = grid1[index] + grid2[index];
    }
}

__kernel void add_grids_image(read_only image2d_t grid1, read_only image2d_t grid2, write_only image2d_t result) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    int2 coord = (int2)(i, j);
    float val1 = read_imagef(grid1, coord).x;
    float val2 = read_imagef(grid2, coord).x;

    float result_val = val1 + val2;

    write_imagef(result, coord, (float4)(result_val, 0.0, 0.0, 0.0));
}
