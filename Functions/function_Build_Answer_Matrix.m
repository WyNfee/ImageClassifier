%this function is convert the 1 dimension answer matrix to two dimension answer matrix
%this is necesary conversion because we are using softmax
%param:
%p_y: the anwser data
%p_output_amount: the output amount of the whole learning network
%return
%r_y:the answer matrix with two dimension, ready for learning algorithm to use
function r_y = function_Build_Answer_Matrix(p_y, p_n_o)
        t_y_m = size(p_y,1);
        t_y = zeros(t_y_m, p_n_o);
        for i = 1 : t_y_m
            t_y(i,p_y(i)) = 1;
        end
        r_y = t_y;
end