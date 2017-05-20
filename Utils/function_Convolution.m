%Compute the Convolution data from input and convolution filter group
%input:
%p_x: the input data set, it should be like:
%         say a input is 3 picture with 20 * 20 resolution (only one
%         channel)
%           the input should be 3 * 400 matrix
%p_f: the input filter group, it should be like:
%       say we have a 6 filter with 3 *3 resolution
%           the filter should be 54 * 1 resolution (9 * 9 * 6)
%p_f_b: the filter bias
%       it should the same size of conved data amount
%            say we convlution 20 * 20 image with 6 3 * 3 resolution filter,
%            it should be 1944 * 1 matrix ( 6 * 18 * 18 )
%p_f_m: the filter amount
%p_f_c: the channels amount
%return:
%r_conv_data: the conved_data, it should be in this form:
%           say we have convoled data with 18 * 18 resolution for 6
%           filters, we have 3 pictures in total, we will have:
%           3 * 1944 (18 * 18 * 6) matrix
function r_conv_data = function_Convolution(p_x, p_f, p_f_b, p_f_m, p_f_c)
    %the total input data amount
    t_m = size(p_x,1);
    %each input data resolution (assume square)
    t_x_s = size(p_x, 2)/p_f_c;
    t_x_d = sqrt(t_x_s);
    
    %the each filter size
    t_f_s = size(p_f, 1) / p_f_m/p_f_c;
    %each filter data resolution
    t_f_d = sqrt(t_f_s);
    
    %each fiter bias size
    t_f_b_s = size(p_f_b, 1) / p_f_m/p_f_c;
    
    %extract the filters for each channel
    t_f_c = reshape(p_f, t_f_s, p_f_m * p_f_c);
    
    %a storage to store all conved data
    t_c_d = [];
    
    %for every input
    for m = 1 : t_m
        %a storage for current cov data
        t_c_c_d = [];
        
        %for each channel
        for j = 1 : p_f_c
        
            %for filter related to this channel
            for i = 1 : p_f_m
                
                %extact the filter from filter storage, and reshape to matrix computable
                s_c_c_f_c = (j-1) * p_f_m + i; %current conv filter column
                s_c_c_f = reshape(t_f_c( :, s_c_c_f_c) , t_f_d,t_f_d); %current conv filter

                %extract the bias, and reshape to matrix for operation
                s_c_c_f_b = reshape(p_f_b( s_c_c_f_c, :), t_f_b_s, t_f_b_s);

                %reorg the input data
                s_c_x = p_x(m,:);
                s_c_x_p = (j - 1) * t_x_s;
                s_c_x = reshape(s_c_x(s_c_x_p+1 : s_c_x_p + t_x_s), t_x_d, t_x_d);

                %convolution
                t_cov = conv2(s_c_x, s_c_c_f, 'valid');

                %add the bias
                t_cov = t_cov + s_c_c_f_b;
                
                %put it in the storage
                t_c_c_d = [t_c_c_d; t_cov(:)];

            end

        end
        %reorg data and put into storage
        t_c_c_d = t_c_c_d';
        
        t_c_d = [t_c_d; t_c_c_d];
        
    end
    
    %return the result
    r_conv_data = t_c_d;
    
end