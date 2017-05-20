%the core learning argorithm of Batch Gradient Desent with Conjunction Gradient Descent
%the neuron activation is using ReLu
%this including both forward propagation and back propagation
%param:
%p_w: the weight
%p_x: the input data
%p_y: the answer
%p_r_p: the regularization param
%p_s_w: the struct of weight
function [r_cost, r_gradient] = function_Compute_Cost_Gradient(p_w, p_x, p_y, p_s_w, p_r_p)
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
    
    %FULL CONNECTION LAYER TO OUTPUT FORWARD PROPAGATION END
    
    %COMPUTE THE LOST OF THE WHOLE NETWORK START
    
    %compute the error without regularization
    t_cost = function_Softmax_Cost(t_softmax, p_y);
    
    %compute the regularization form
    %regularization form for layer one:
    t_w_h_r = function_L2_Weight_Regularization(t_w_h, t_m, p_r_p);
    %regularization form for layer two:
    t_w_o_r = function_L2_Weight_Regularization(t_w_o, t_m, p_r_p);
    
    t_cost = t_cost + t_w_h_r + t_w_o_r;
    
    r_cost = t_cost;
    
    % COMPUTE THE LOST OF THE WHOLE NETWORK END
    
    %FORWARD PROPAGATION END
    
    
    %BACK PROPAGATION START
    
    %COMPUTE FULL CONNECTION TO OUTPUT LAYER ERROR AND GRADIENT START
    %E4/Z4 = E4/softmax * /softmax*Z4 =y-answer
    t_delta_o = t_softmax - p_y;
   
    %compute the gradient of layer two weight
    %use chain rule to compute: 
    %E3/w2 = E3/a3 * a3/z3 * z3/w2 = 1 * g' * a2 (compute order is not
    %considered)
    t_w_o_grad = t_delta_o' * t_a_h / t_m;
    %the bias donot need the regularization
    t_w_o_grad_reg = ones(size(t_w_o));
    t_w_o_grad_reg(:, 1) = 0;
    t_w_o_grad_reg = t_w_o_grad_reg .* t_w_o * p_r_p / t_m;
    t_w_o_grad = t_w_o_grad + t_w_o_grad_reg;
    %COMPUTE FULL CONNECTION TO OUTPUT LAYER ERROR AND GRADIENT END
    
    %COMPUTE FULL CONNECTION TO HIDDEN LAYER ERROR AND GRADIENT START
    %compute the layer 3 error
    t_delta_h = t_delta_o * t_w_o;
    %compute teh gradient of layer one weight
    %use chain rule as well to compute, but this time, we have to separete
    %the computation, because the hidden layer got 1 addional column, and
    %it shouldnot be put into the gradient
    %transit error to layer one z form E2/Z1 = E2/A2 * A2/Z1
    %but the original output haven't added 1 column, so we add it back
    t_z_h = [t_helper, t_z_h];
    %use the changed output data to compute gradient
    t_delta_h = t_delta_h .* function_ReLu_Gradient(t_z_h);
    %remove the addtional comlumn
    t_delta_h = t_delta_h(:,(2:size(t_delta_h ,2)));
    %continue to compute E2/W1 = E2/Z1 * Z1/W1
    t_w_h_grad = t_delta_h' *t_a_c  / t_m;
    
    %the bias donot need to regularization
    t_w_h_grad_reg = ones(size(t_w_h));
    t_w_h_grad_reg(:,1) = 0;
    t_w_h_grad_reg = t_w_h_grad_reg .* t_w_h * p_r_p / t_m;
    t_w_h_grad = t_w_h_grad + t_w_h_grad_reg;
    %COMPUTE FULL CONNECTION TO HIDDEN LAYER ERROR AND GRADIENT END
    
    %COMPUTE THE CONV FILTER ERROR AND GRADIENT START
    %compute the layer 2 error
    t_delta_c = t_delta_h * t_w_h;
    %remove the bias column
    t_delta_c = t_delta_c(:,(2:size(t_delta_c,2)));
    
    t_delta_c = function_MaxPooling2x2Back(t_delta_c, p_s_w.t_n_k, p_s_w.t_n_c);
    
    %compute the error
    t_delta_c = t_delta_c .* function_ReLu_Gradient(t_z_c);


    %Use Conv Operation to compute the grad of conv filter
    t_w_c_grad = function_Convolution_Gradient(t_delta_c, p_x, p_s_w.t_n_k, p_s_w.t_n_c);
 
    %the bias do not need any grad
    t_w_c_b_grad = zeros(s_n_k, 1);
    
    %COMPUTE THE CONV FILTER ERROR AND GRADIENT END
    
    %pack the gradient again
    r_gradient = [t_w_c_grad(:); t_w_c_b_grad(:)];
    r_gradient = [r_gradient(:); t_w_h_grad(:)];
    r_gradient = [r_gradient(:) ; t_w_o_grad(:)];
    
end