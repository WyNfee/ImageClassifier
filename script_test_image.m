%clear and close everything
clear; close all; clc;

load('learnt_weight.mat');
load('training_mu_var.mat');

addpath(genpath('Data'))
addpath(genpath('Utils'))
addpath(genpath('Functions'))
addpath(genpath('Network'))

t_n_i = 3072; %input neuron
t_n_c = 3; % the channel amount
t_n_k = 6; % the kernel amount for each channel
t_s_k = 3; % the size of each kernel
t_n_h = 1000;%the hidden neuron amount
t_n_o = 10;% the output neuron amount

t_n_p = ((sqrt(t_n_i / t_n_c) - t_s_k + 1 ) / 2).^2 * t_n_k * t_n_c; %the neuron amount after pooling process

%put information for unpack here, the struct of weight
t_s_w = struct(...
    't_n_c',t_n_c,...% amount of channel
    't_n_k',t_n_k,...%amount of kernel
    't_s_k', t_s_k,...%size of kernel
    't_n_p', t_n_p,...%amount of neuron after pooling 
    't_n_h', t_n_h,...%the amount neuron of hidden
    't_n_o', t_n_o...%the amount neuron of output
    );

image = imread('8.jpg');

image(:,:,1) = image(:,:,1)';
image(:,:,2) = image(:,:,2)';
image(:,:,3) = image(:,:,3)';

imagesc(image);
axis image off;

image_data = double(image(:)');

image_data = (image_data - t_training_mu) ./sqrt(t_training_var);

data_label = ["airplane";"automobile";"bird";"cat";"deer";"dog";"frog";"horse";"ship";"truck"];

predict_matrix = function_Do_Prediction(image_data,t_l_w, t_s_w);

[v, i] = max(predict_matrix);

data_label(i)








