%the core learning algorithm locate
%param:
%p_x: the training data set
%p_y: the training answer
%output
%r_w: the weight learnt in this training data set
%r_c: the cost history
%r_n: the network params
function[r_w, r_c, r_n] = function_Learning_Algorithm(p_x, p_y)
    %setup the network structure
    t_n_i = 3072; %input neuron
    t_n_c = 3; % the channel amount
    t_n_k = 6; % the kernel amount for each channel
    t_s_k = 3; % the size of each kernel
    t_n_h = 1000;%the hidden neuron amount
    t_n_o = 10;% the output neuron amount
    
    %some parameters extract from input data
    t_x_m = size(p_x, 1);%the input data amount
    t_y = function_Build_Answer_Matrix(p_y,t_n_o);%build the answer matrix for leanring algorithm
    
    %WEIGHT INITIALIZATION STARTS
    
    %init kernel and wrap to a kernel package
    s_n_k = t_n_k * t_n_c; %all kernels we have
    s_s_k = t_s_k .^ 2;
    t_w_k = zeros(s_s_k , s_n_k);
    for i = 1 : s_n_k
        t_w_k(:, i) = function_XavierInitialization_For_ReLu(1, s_s_k);
    end
    t_w_k = t_w_k(:); % the kernel package structure is (s_n_k * s_s_k : 1) matrix, each s_s_k is a kernel
    %the last s_n_k weights are used for bias
    s_w_b = rand(s_n_k,1); %we don't use normal distrubtion for bias init, we expect a uniform distribution here
    t_w_k = [t_w_k; s_w_b]; %weight for convolution layer complete
    
  
    %init the weight for full connection to hidden layer after pooling process
    t_n_p = ((sqrt(t_n_i / t_n_c) - t_s_k + 1 ) / 2).^2 * t_n_k * t_n_c; %the neuron amount after pooling process
    t_w_h = function_XavierInitialization_For_ReLu(t_n_p + 1, t_n_h); %init the weight for full connection layer
    
    %init the weight for full connection to output layer
    t_w_o = function_XavierInitialization_For_ReLu(t_n_h + 1, t_n_o);
    
    %Pack the weight in 1 container
    t_w = [t_w_k(:); t_w_h(:)];
    t_w = [t_w(:); t_w_o(:)];
    
    %put information for unpack here, the struct of weight
    t_s_w = struct(...
        't_n_c',t_n_c,...% amount of channel
        't_n_k',t_n_k,...%amount of kernel
        't_s_k', t_s_k,...%size of kernel
        't_n_p', t_n_p,...%amount of neuron after pooling 
        't_n_h', t_n_h,...%the amount neuron of hidden
        't_n_o', t_n_o...%the amount neuron of output
        );
    
    %WEIGHT INITIALIZATION ENDS
    
    %CORE TRAINING PROCESS STARTS
    
    %set up hyper parameters for machine learning
    t_s_h_p = struct(...
        't_l_r', 0.01, ...%the learning rate
        't_l_r_d_f', 2100,...%the learning rate decay frequency
        't_l_r_d_r', 0.8, ...%the learning rate decay rate
        't_r_p', 0.1,... %the L2 regularization param
        't_s_g_d', 50, ...%the sgd size
        't_i_t', 100000, ...%the iteration times
        't_r_f', 100 ...%the record frequency
        );
    
   %the learnt weight
    t_l_w = t_w;
    
    %do iterations
    for i = 1 : t_s_h_p.t_i_t
        
        %picked index
        s_p_i = floor(rand(1, t_s_h_p.t_s_g_d) * (t_x_m - 1) ) + 1;
       	s_p_x = p_x(s_p_i(1:end), :);
        s_p_y = t_y(s_p_i(1:end), :);
        
        [t_cost_param, t_gradient_param] = function_Compute_Cost_Gradient(t_l_w,s_p_x, s_p_y, t_s_w, t_s_h_p.t_r_p);
        
        %simply compute the gradient
        t_l_w = t_l_w - t_s_h_p.t_l_r * t_gradient_param;
        
        %decay the learning rate
        if (rem( i, t_s_h_p.t_l_r_d_f) == 0)
            t_s_h_p.t_l_r = t_s_h_p.t_l_r * t_s_h_p.t_l_r_d_r;
        end
            
        %output the cost to console every 100 iterate, so that we know
        %whether it is working, and the progress so far
        fprintf('update cost, current cost %.6f,\n',t_cost_param);
        
    end

    %CORE TRAINING PROCESS ENDS
end
  