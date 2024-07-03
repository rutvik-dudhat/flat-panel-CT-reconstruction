__kernel void backproject(
    read_only image2d_t sinogram,
    write_only image2d_t reco,
    int num_projections,
    int detector_size,
    float angular_increment_degree,
    float detector_spacing,
    float detector_origin,
    int reco_sizeX,
    int reco_sizeY,
    float reco_originX,
    float reco_originY,
    float reco_spacingX,
    float reco_spacingY)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < reco_sizeX && y < reco_sizeY) {
        float reco_x = reco_originX + x * reco_spacingX;
        float reco_y = reco_originY + y * reco_spacingY;
        float sum = 0.0f;

        for (int p = 0; p < num_projections; ++p) {
            float angle = p * angular_increment_degree;
            float cos_angle = cos(angle);
            float sin_angle = sin(angle);

            float detector_x = reco_x * cos_angle + reco_y * sin_angle;
            int detector_idx = round((detector_x - detector_origin) / detector_spacing);

            if (detector_idx >= 0 && detector_idx < detector_size) {
                int2 coord = (int2)(detector_idx, p);
                sum += read_imagef(sinogram, coord).x;
            }
        }

        sum *= angular_increment_degree;
        write_imagef(reco, (int2)(x, y), (float4)(sum, 0.0f, 0.0f, 0.0f));
    }
}
