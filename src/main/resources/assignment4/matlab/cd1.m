function ret = cd1(rbm_w, visible_data)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.
    %error('not yet implemented');

     visible_data = sample_bernoulli(visible_data);
     % fprintf(' cd1 step 1:  %f\n', sum(visible_data(:)));
     % fprintf(' cd1 step 1a1:  %f\n', sum(rbm_w(:)));
     step1a = visible_state_to_hidden_probabilities(rbm_w,visible_data);
     % fprintf(' cd1 step 1a2:  %f\n', sum(step1a(:)));
    h1 = sample_bernoulli(step1a);
    % fprintf('cd1 step 2:  %f\n', sum(h1(:)));
    vrc = sample_bernoulli(hidden_state_to_visible_probabilities(rbm_w,h1));
    % fprintf('cd1 step 3:  %f\n', sum(vrc(:)));
    h2 = visible_state_to_hidden_probabilities(rbm_w,vrc);
    % fprintf('cd1 step 4:  %f\n', sum(h2(:)));
    x1 = configuration_goodness_gradient(visible_data, h1);
 	% fprintf('cd1 step 5:  %f\n', sum(x1(:)));
 	% fprintf('cd1 step 5a:  %f\n', sum(visible_data(:)));
    x2 = configuration_goodness_gradient(vrc, h2);
     % fprintf('cd1 step 6:  %f\n', sum(x2(:)));
    ret =  x1 - x2;
end
