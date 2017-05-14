%clear and close everything
clear; close all; clc;

%add searching path 
addpath(genpath('Data'))
addpath(genpath('Utils'))
addpath(genpath('Functions'))

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
    
    %TODO: display the label data in training cases
    function_Preview_Image(g_all_original_data, g_all_training_data, g_all_answer, g_h_data_channel, g_labels_string);
    
else
    fprintf('Skip image preview\n');
end
g_data_amount = size(g_all_training_data);



