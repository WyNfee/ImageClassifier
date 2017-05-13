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
function_Prepare_Data(g_h_data_res, g_h_data_channel);
