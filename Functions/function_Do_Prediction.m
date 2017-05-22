%This fucntion do prediction on learnt data
%
function [r_prediction_matrix] = function_Do_Prediction...
    (...
    p_x, p_w, ...
    p_s_w...
    )
    %UNPACK THE WEIGHT STARTS
    
    %extact all kernels params
    s_s_k = (p_s_w.t_s_k).^2; %each kernel size
    s_n_k = p_s_w.t_n_k * p_s_w.t_n_c;%kernel amount
    %extract kernel weight
    s_w_k_p = s_s_k * s_n_k;%all data amount related to kernel weight
    t_w_k = p_w(1 : s_w_k_p);%extract them
    %extract kernel bias
    t_w_b = p_w(s_w_k_p + 1: s_w_k_p + s_n_k);
    
    %extact all weights for hidden layers
    s_w_h_p = s_w_k_p + s_n_k;
    t_n_w_h = p_s_w.t_n_h * (p_s_w.t_n_p + 1 );
    t_w_h = p_w(s_w_h_p + 1 : s_w_h_p + t_n_w_h);
    t_w_h = reshape(t_w_h, p_s_w.t_n_h, p_s_w.t_n_p+1);
    
    %extract all weights for output layers
    s_w_o_p = s_w_h_p + t_n_w_h;
    t_w_o = p_w(s_w_o_p + 1 : end);
    t_w_o = reshape(t_w_o, p_s_w.t_n_o, p_s_w.t_n_h + 1);
    
    %UNPACK THE WEIGHT ENDS
    
    %create a input helper
    t_m = size(p_x, 1);
    t_helper = ones(t_m,1);
    
    %FORWARD PROPAGATION STARTS
    
    %CONVOLUTION FORWARD PROPAGATE STARTS    
    t_z_c = function_Convolution(p_x, t_w_k, t_w_b, p_s_w.t_n_k, p_s_w.t_n_c);
    t_a_c = function_ReLu(t_z_c);
    
    t_a_c = function_MaxPooling2x2(t_a_c, p_s_w.t_n_k, p_s_w.t_n_c);
    
    %CONVOLUTION FORWARD PROPAGATION END
    
    %FULL CONNECTION LAYER TO HIDDEN FORWARD PROPAGATION START
    
    %prepare the layer one input data
    %add additonal 1 column at the begining 
    t_a_c = [t_helper, t_a_c];
    
    %the layer one output
    t_z_h = t_a_c *  t_w_h';
        
    %the layer two input
    %add additonal 1 column at the begining
    t_a_h = function_ReLu(t_z_h);
    
    %FULL CONNECTION LAYER TO HIDDEN FORWARD PROPAGATION END
    
    %FULL CONNECTION LAYER TO OUTPUT FORWARD PROPAGATION START
    
    t_a_h  = [t_helper, t_a_h];

    %the layer two output
    t_z_o = t_a_h * t_w_o';
    
    %the prediction, the layer 3 data
    t_softmax = function_Softmax(t_z_o);
    t_predictions_matrix = t_softmax';
    
    %return the prediction
    r_prediction_matrix = t_predictions_matrix;
    

end