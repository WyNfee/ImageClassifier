%This function check the performance of learning algorithms
%p_x: the input data
%p_y: the input answer
%p_w: the weight used for prediction
%p_s_n: the structure of the network
function function_Check_Performance(p_x, p_y, p_w, p_s_n)
    

   %Do prediction
   %the prediction matrix (soft max)
   t_p_m = function_Do_Prediction(p_x, p_w, p_s_n);
    %we use the max probability in k-means output, in practise, sometimes using
    %top 5 output, this cases is so small, using top 5 is silly
    
    [t_probability, t_prediction] = max(t_p_m);

    %output the result
    t_right_prediction_count = sum(t_prediction' == p_y);
    t_accuracy = t_right_prediction_count / size(p_y,1);

    fprintf('prediction accurracy %1.6f\n',t_accuracy)
    
    %sort result and reort index
    [t_s_r, t_s_i] = sort(t_p_m, 1, 'descend');
    
    t_x_m = size(p_x, 1);
    
    %top 3 accuracy count
    t_t3_a_c = 0;
    
    for i = 1 : t_x_m
        
        for j = 1 : 3
            
            if(t_s_i(j, i) == p_y(i))
                t_t3_a_c = t_t3_a_c + 1;
                break;
            end
            
        end

    end
    
    t_t3_a = t_t3_a_c / t_x_m;
    
    fprintf('prediction top 3 accurracy %1.6f\n',t_t3_a);
    
end