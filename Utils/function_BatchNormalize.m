%computation the batch normalization for data
%input:
%p_x: the data need to batch normalization
%p_s_g: the shift gamma
%p_s_b: the shift bias
%output
%the data after batch normalization
function [r_bn, r_mu, r_var] = function_BatchNormalize(p_x, p_s_g, p_s_b)
    t_x = double(p_x);
    t_epsilon = 1e-8;
    %the amount of p_x
    t_x_m = size(t_x, 1);
    %the mean of each features
    t_x_mu = sum(t_x, 1) ./ t_x_m;
    %the variance of each features
    t_x_var = sum((t_x - t_x_mu).^2, 1) / t_x_m;
    %the raw bn data
    t_x_hat = (t_x - t_x_mu) ./ sqrt(t_x_var + t_epsilon);
    %output the shift bn 
    r_bn = p_s_g .* t_x_hat + p_s_b;
    r_mu = t_x_mu;
    r_var = t_x_var;
    
end