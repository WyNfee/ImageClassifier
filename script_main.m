%clear and close everything
clear; close all; clc;

%add searching path 
addpath(genpath('Data'))
addpath(genpath('Utils'))
addpath(genpath('Functions'))

%Provide some hypter parameter here
g_h_data_res = 32;%the resolution of each data
g_h_data_channel =3; %the channel of each data

%read the CFIAR data
[g_all_original_data, g_all_training_data, g_all_answer] =function_Read_Data_Files();

%preview the data

s = input('preview the image data?, y to yes:','s');
if(s == 'y')
    fprintf('Preview the image\n');
    
    function_Preview_Image(g_all_original_data, g_all_training_data, g_h_data_channel);
    
else
    fprintf('Skip image preview\n');
end
g_data_amount = size(g_all_training_data);



