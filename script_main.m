%clear and close everything
clear; close all; clc;

%add searching path 
addpath(genpath('Data'))
addpath(genpath('Utils'))
addpath(genpath('Functions'))
addpath(genpath('Network'))

%read the CFIAR data
[...
    t_training_data, t_training_answer,...
    t_test_data, t_test_answer,...
    t_labels_string...
] =function_Read_Data_Files();

%preview the image
function_Preview_Image(t_training_data,t_training_answer,t_labels_string);

%learning algorithm
function_Learning_Algorithm(t_training_data, t_training_answer);