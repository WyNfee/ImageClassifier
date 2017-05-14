%clear and close everything
clear; close all; clc;

%add searching path 
addpath(genpath('Data'))
addpath(genpath('Utils'))
addpath(genpath('Functions'))
addpath(genpath('Network'))

%Provide some hypter parameter here
g_h_data_res = 32;%the resolution of each data
g_h_data_channel =3; %the channel of each data

g_read_from_data_file = false;

if(g_read_from_data_file == true)
    %read the CFIAR data
    [g_all_original_data, g_all_training_data, g_all_answer, g_labels_string] =function_Read_Data_Files();

    s = input('store the save data?, y to yes:','s');
    if( s== 'y')
        save('data_stored_training_data.mat','g_all_original_data','g_all_training_data','g_all_answer','g_labels_string');
        fprintf('Data Saved\n');
    else
        fprintf('No Data Saved\n');
    end
else
    %load preprocessed data
    load('data_stored_training_data.mat');
end

%preview the data

s = input('preview the image data?, y to yes:','s');
if(s == 'y')
    fprintf('Preview the image\n');
    
    function_Preview_Image(g_all_original_data, g_all_training_data, g_all_answer, g_h_data_channel, g_labels_string);
    
else
    fprintf('Skip image preview\n');
end

%Do a simple machine learning alogrithm

%setup machine learning architecture
g_layer_input_neuron_amount = 3072;%input neuron amount 3072 = 32 * 32 * 3
g_layer_hidden_neuron_amount = 1024; %hidden neuron amount 1024;
g_layer_output_neuron_amount = 10; %classifer amount 10 (10 classifiers)

%first; do weigth initialization, and pack them together for learning
%algorithm
t_layer_input_weight = function_XavierInitialization_For_ReLu(g_layer_input_neuron_amount + 1, g_layer_hidden_neuron_amount);
t_layer_hidden_weight = function_XavierInitialization_For_ReLu(g_layer_hidden_neuron_amount + 1, g_layer_output_neuron_amount);
t_packed_weight = [t_layer_input_weight(:); t_layer_hidden_weight(:)];

%second, organize the answer data
%the reason of organize data is because using k-means, the result we get is
%not only 1 answer, but 10 answers
t_answer_data_size = size(g_all_answer, 1);
g_answer_data = zeros(t_answer_data_size, g_layer_output_neuron_amount);
for i = 1 : t_answer_data_size
    g_answer_data(i, g_all_answer(i)) = 1;
end

%third: set up parameters for learning algorithm
%1. provide the size of weight so that we can unpack it in learning
%algorithm
t_layer_input_weight_size = size(t_layer_input_weight);
t_layer_hidden_weight_size = size(t_layer_hidden_weight);

%2. regularization form (0 to close regularization)
t_h_reularization_param = 0;

%3. set the learning rate; default, 0.01;
t_h_learning_rate = 0.01;

%4. SGD batch amount, default 500;
t_sgd_data_size = 50;

%5. SGD iteration time; do 500,00 iterations (need to adjust later)
t_iteration_time = 50000;

%6. cost data storage for ploting
t_plot_frequency = 100; %1,000 times per plot
t_cost_data_record = zeros(t_iteration_time/t_plot_frequency, 1);

%Forth. machine learning architecture related parameter
%1. Adam parameter
t_adam_param_epsilon = 1e-7;
t_adam_param_beta1 = 0.9;
t_adam_param_beta2 = 0.999;
t_adam_weight_movement = zeros(size(t_packed_weight));
t_adam_weight_velocity = zeros(size(t_packed_weight));

g_do_gradient_descent = true;

if(g_do_gradient_descent == true)
    
    %the core learning process
    for i = 1 : t_iteration_time
        %pick random indexes from training data set, choose the random data
        %the reason of writing like this: make sure the index we are picked
        %within [1, t_answer_data_size];
        t_rand_picked_data_index = floor(rand(1, t_sgd_data_size) * (t_answer_data_size -1 )) + 1;
        %pick the data and answer data
        t_rand_picked_data  = g_all_training_data(t_rand_picked_data_index(1:end), :);
        t_rand_picked_answer = g_answer_data (t_rand_picked_data_index(1:end), :);
        
        [t_learnt_cost, t_learnt_gradient] = function_NN_Learning_Algorithm(t_packed_weight,t_rand_picked_data, t_rand_picked_answer,...
            t_layer_input_weight_size, t_layer_hidden_weight_size, t_h_reularization_param);
        
        %compute two main update parameter for adam
        %t_adam_weight_movement = t_adam_param_beta1 .* t_adam_weight_movement + ( 1- t_adam_param_beta1) .* t_learnt_gradient;
        %t_adam_weight_velocity = t_adam_param_beta2 .* t_adam_weight_velocity + ( 1 - t_adam_param_beta2) .* (t_learnt_gradient.^2);
        
       %correct the bias to boost up the parameter, this should not update
        %the original parameter
       % t_adam_weight_movement_updater = t_adam_weight_movement ./ (1 - t_adam_param_beta1.^i);
        %t_adam_weight_velocity_updater = t_adam_weight_velocity ./ ( 1- t_adam_param_beta2.^i);
        
        %compute the updater
        %t_adam_updater = t_h_learning_rate .* t_adam_weight_movement_updater ./ (sqrt(t_adam_weight_velocity_updater) + t_adam_param_epsilon);
        
        %do the update
        %t_packed_weight = t_packed_weight - t_adam_updater;
        t_packed_weight = t_packed_weight - t_learnt_gradient;
        
        if( rem(i, t_plot_frequency) == 0)
            t_cost_data_record(i/t_plot_frequency) = t_learnt_cost;
           
        end
        
         fprintf('update cost, current cost %.6f,\n',t_learnt_cost);
        
    end
    
    %Save the data so taht we can use it to generate the plot
    %because running the learning alogorithm is taking time
    %we save the data can saving the time running it again
    s = input('save the loss data? If close the training gate, it can run again using the loss data and will be able to create plot?, y to save:','s');
    
    if(s == 'y')
        save('data_image_classifier.mat', 't_packed_weight', 't_cost_data_record');
        fprintf('Data Saved\n');
    else
        fprintf('No Data Saved\n');
    end

else
    
    %we can do plot and check the performance without running the learning
    %algorithm again
    
    %plot the gradient descent
    load('data_image_classifier.mat');
    
    %prepare the data for plot
    t_cost_data_size = length(t_cost_data_record);
    t_cost_data = zeros(t_cost_data_size, 2);
    
    for i = 1 : t_cost_data_size
        
        t_cost_data(i, 1) = i;
        t_cost_data(i, 2) = t_record_cost_data(i);
        
    end
    
    %plot the data
    plot(t_cost_data(:,1), t_cost_data(:,2),'r-');
    
    %Save the data to compare with other learning algorithm
    s = input('save the plot data?, y to save:','s');
    if(s == 'y')
        save('data_lost_image_classifier.mat', 't_cost_data');
        fprintf('Data Saved\n');
    else
         fprintf('No Data Saved\n');
    end
    
end

