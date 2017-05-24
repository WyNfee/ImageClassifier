%This function will read the content from the data file
%output
%r_pre_process_data_set: the data set has been pre-processsed
%r_answer_set: the whole answer set
function ...
    [...
    r_training_data_set,r_traning_answer_set,...
    r_test_dataset,r_test_answer_set,...
    r_data_label,...
    r_mu, r_var...
    ] = function_Read_Data_Files()
    %init a temporary data set
    t_pre_process_data_set = [];
    t_answer_set=[];
    
    %Process the training data first
    %hard code the index here, the data file locates in Data directory
    for i = 1 : 1 % load only 1 pack for general testing
        t_data_file_name = sprintf('data_batch_%d.mat', i);
        t_data_file = load(t_data_file_name);
        t_data = t_data_file.data;
        t_answer = t_data_file.labels + 1;
        
        %preprocess read data
        [t_pre_process_data, t_pre_process_data_mu, t_pre_process_var] = function_Preprocess_Data(t_data);
        
        %put the prepared data into return blocks
        t_answer_set = [t_answer_set; t_answer];
        t_pre_process_data_set = [t_pre_process_data_set; t_pre_process_data];
    end
    
    r_traning_answer_set = t_answer_set;
    r_training_data_set = t_pre_process_data_set;
    r_data_label = ["airplane";"automobile";"bird";"cat";"deer";"dog";"frog";"horse";"ship";"truck"];
    r_mu = t_pre_process_data_mu;
    r_var = t_pre_process_var;
    
    %Process the test data
    t_data_file = load('test_batch.mat');
    t_data = t_data_file.data;
    t_answer = t_data_file.labels +1;
    
    r_test_dataset = (double(t_data) - t_pre_process_data_mu) ./ sqrt(t_pre_process_var);
    r_test_answer_set = t_answer;
    
end