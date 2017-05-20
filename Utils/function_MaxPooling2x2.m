%This function is doing pooling operation, by 2 * 2
%input
%p_x: the data need to do pooling
%p_n_f: the filter amount
%p_n_c: the channel amount
%output:
%r_p: the data after pooling
function r_pooling = function_MaxPooling2x2(p_x, p_n_f, p_n_c)

    %the size of the input x;
    t_m = size(p_x, 1);
    %size of each image
    t_x_s = size(p_x,2) / p_n_f / p_n_c;
    %the dimension of each image
    t_x_d = sqrt(t_x_s);
    %the size of each channel
    t_x_s_c = t_x_s * p_n_f;
    
    %a storage for pooled data
    t_p_d = [];
    
    for m = 1 : t_m
        %current data
        s_c_d = p_x(m, :);
        
        %a storage for current data
        t_c_p_d=[];
        
        %for each channel
        for j = 1 : p_n_c
            
            %for each filter
            for i = 1 : p_n_f
                
                %current data position
                t_c_d_p = (j - 1) * t_x_s_c + (i - 1) * t_x_s;

                %grab the current data
                t_c_data = s_c_d( (t_c_d_p + 1) : (t_c_d_p + t_x_s));
                t_c_data = reshape( t_c_data, t_x_d,t_x_d);

                %do pooling
                %iteration indicator
                t_iter = floor((t_x_d+1)/2);

                %current pooling
                t_c_p = zeros(t_iter, t_iter);

                %height index
               for h = 1 : t_iter
                   %width index
                   for w = 1 : t_iter

                       %compute each index
                       t_h = (h -1) * 2;
                       t_w = (w - 1) * 2;

                       %reshape to image
                       t_p_data = reshape(t_c_data (t_w + 1 : t_w+2, t_h+1:t_h+2), 2, 2);
                       %do pooling
                       t_c_p(w, h) = max(max(t_p_data));


                   end
               end
               
               %store the pooling data
               t_c_p_d = [t_c_p_d; t_c_p(:)];   
               
            end
        
        end
        
        t_c_p_d = t_c_p_d';
        t_p_d = [t_p_d; t_c_p_d];
    end
    
    
    r_pooling = t_p_d;
end