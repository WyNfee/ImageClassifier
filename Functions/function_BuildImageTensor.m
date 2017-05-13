function r_image_tensor =function_BuildImageTensor(p_original_image_data, p_image_channels)
    %build the image matrix
        t_image_data_size = size(p_original_image_data, 2);
        t_image_channel_size = t_image_data_size / p_image_channels;
        t_image_res = sqrt(t_image_channel_size);
        
        %split the image data into channels
        t_image_data_start = 1;
        t_image_data_end = t_image_channel_size;
        t_image_channel_r = reshape(p_original_image_data(t_image_data_start:t_image_data_end), t_image_res, t_image_res);
        
        t_image_data_start = t_image_data_end + 1;
        t_image_data_end = t_image_data_end + t_image_channel_size;
        t_image_channel_g = reshape(p_original_image_data(t_image_data_start:t_image_data_end), t_image_res, t_image_res);
        
        t_image_data_start = t_image_data_end + 1;
        t_image_data_end = t_image_data_end + t_image_channel_size;        
        t_image_channel_b = reshape(p_original_image_data(t_image_data_start:t_image_data_end), t_image_res, t_image_res);
        
        %combine the image channel as a image tensor
        t_image_tensor(:,:,1) = t_image_channel_r';
        t_image_tensor(:,:,2) = t_image_channel_g';
        t_image_tensor(:,:,3) = t_image_channel_b';
        
        r_image_tensor = t_image_tensor;
end