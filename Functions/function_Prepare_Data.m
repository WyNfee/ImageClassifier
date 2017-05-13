%prepare the data set for processing
%data set may not be normalized
%we need to make them suitable for further computation
function [r_prepared_data_set, r_answer_set] = function_Prepare_Data(p_input_res, p_input_channel)
    %read the data files
    [t_data_set, t_answer_set] = function_Read_Data_Files();
    %now we have got the answer data
    r_answer_set = t_answer_set;
    
end