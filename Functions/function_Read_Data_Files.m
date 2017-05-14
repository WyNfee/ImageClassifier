%This function will read the content from the data file
%output
%r_data_set: the whole data set
%r_pre_process_data_set: the data set has been pre-processsed
%r_answer_set: the whole answer set
function [r_data_set, r_pre_process_data_set,r_answer_set, r_data_label] = function_Read_Data_Files()
    %init a temporary data set
    t_data_set = [];
    t_pre_process_data_set = [];
    t_answer_set=[];
    
    %hard code the index here, the data file locates in Data directory
    for i = 1 : 5
        t_data_file_name = sprintf('data_batch_%d.mat', i);
        t_data_file = load(t_data_file_name);
        t_data = t_data_file.data;
        t_answer = t_data_file.labels + 1;
        
        %preprocess read data
        t_pre_process_data = function_Preprocess_data(t_data);
        
        %put the prepared data into return blocks
        t_data_set = [t_data_set; t_data];
        t_answer_set = [t_answer_set; t_answer];
        t_pre_process_data_set = [t_pre_process_data_set; t_pre_process_data];
    end
    
    r_data_set = t_data_set;
    r_answer_set = t_answer_set;
    r_pre_process_data_set = t_pre_process_data_set;
    r_data_label = ["airplane";"automobile";"bird";"cat";"deer";"dog";"frog";"horse";"ship";"truck"];
end