%This function do upsampling operation, revert the image back from pooling
%input:
%p_x: the input of matrix need to revert/upsampling
%p_n_f; the filter amount
%p_n_c: the channel amount
%output
%r_upsampling: the upsampled data
function r_upsampling = function_MaxPooling2x2Back(p_x, p_n_f, p_n_c)
    %the data amount
    t_m = size(p_x, 1);
    t_x_c = size(p_x, 2)/p_n_c;
    t_x_s = t_x_c/p_n_f;
    t_x_d = sqrt(t_x_s);
    
    %the storage for upsample data
    t_u = [];
    
    for m = 1 : t_m
        %for current up sampling data
        t_c_u = [];
        
        t_c_u_d = p_x(m, :);
        
        %for each channel
        for j = 1 : p_n_c
            %for each kenel
            for i = 1 : p_n_f
            
                %current data position
                t_c_d_p = (j-1) * t_x_c + (i-1) * t_x_s;

                %current data
                t_c_d = reshape(t_c_u_d(t_c_d_p + 1 : t_c_d_p + t_x_s), t_x_d, t_x_d);

                %current up sampling data
                t_c_u_s_d = zeros(t_x_d * 2, t_x_d * 2);

                for h = 1 : t_x_d
                    for w = 1 : t_x_d

                        t_elment = t_c_d(h, w);

                        t_h = h * 2 - 1;
                        t_w = w * 2 -1;

                        %up sampling
                        t_c_u_s_d(t_h : t_h+1, t_w : t_w+1) = t_elment;

                    end
                end
                %store the sample data
            t_c_u = [t_c_u; t_c_u_s_d(:)];
            end
        
        end
        
        %put it to upsample container
        t_c_u = t_c_u';
        t_u = [t_u; t_c_u];
        
    end
    
    r_upsampling = t_u;
end