%This function is used to preprocess the data for image machine leanring
%the idea is: center and normalize
%param:
%p_input_data: the input data to do preprocessing
%return:
%r_data: the data used for image processing
function [r_data, r_mu, r_var] = function_Preprocess_Data(p_x)
    
    %for raw data processing, we don't do any shift
    [r_data, r_mu, r_var] = function_BatchNormalize(p_x, 1, 0);

end