%This function will compute the gradient of a convolution process
%input
%p_d: the delta (error to input)
%       this must have been computed out before passing through this
%       function.
%       if we have a covolution input, 3 images with 18 * 18 conved data,
%       each data applies 6 filters
%       the data should be 3 * 1944 (18*18*6) matrix
%   
%p_x: the input data of the current layer
%       if the input of the conved data is 3 image with 20 * 20 resolution
%       then the data should be 3 * 400 (20 * 20) matrix
%p_n_f: filter amount, in comments space, it is 6
%p_n_c: 
%output
%r_grad:
%       the gradient compute based on these parameters
%       in comment space, the filter should be 6 filters with 3 * 3
%       resolutions
function r_grad = function_Convolution_Gradient(p_d, p_x, p_n_f, p_n_c)
    
    % the amount of input;
    t_m = size(p_x, 1);
    %the data size per channel
    t_x_c = size(p_x, 2) / p_n_c;
    %data dimension per dimension
    t_x_d = sqrt(t_x_c);
    
    % the size of each delta error per channel
    t_d_c = size(p_d, 2) / p_n_c;
    % the size of each error per filter
    t_d_s = t_d_c /  p_n_f;
    % the delta error for each filter
    t_d_d = sqrt(t_d_s);
    
    %the convolution filter dimension
    t_f_d = t_x_d - t_d_d + 1;
    
    %Worthy to note here, 
    %when computing the gradient, we revert the loop
    %process from compute the convolution, 
    %the reason is we need to compute the gradient for every filter, not
    %every item
   
    %a storage for gradient
    t_g = [];
   
    %for every channel
    for j = 1 : p_n_c
        %for every filter
        for i = 1 : p_n_f
        
            %a storage for current filter gradient
            %filter grad
            t_f_g = zeros(t_f_d, t_f_d);

            %for every input
            for m = 1 : t_m
                %current delta pos
                t_c_d_p = (j - 1) * t_d_c + (i - 1) * t_d_s;

                %current delta for this filter
                t_c_d = p_d(m,  (t_c_d_p + 1 : t_c_d_p + t_d_s));
                t_c_d = reshape(t_c_d, t_d_d, t_d_d);

                %re-org original data
                t_c_x_p = (j - 1) * t_x_c;
                t_c_x = p_x(t_c_x_p + 1 : t_c_x_p + t_x_c);
                t_c_x = reshape(t_c_x, t_x_d, t_x_d);

                %compute current filter grad
                t_c_f_g = conv2(t_c_x, rot90(t_c_d, 2), 'valid');

                %add all grad
                t_f_g = t_f_g + t_c_f_g;

            end
            %compute the final grad
            t_f_g = t_f_g ./ t_m;
            %put it into the storage

            t_g = [t_g; t_f_g(:)];
        end
        
    end

    % give it to output
    r_grad = t_g;
    
end