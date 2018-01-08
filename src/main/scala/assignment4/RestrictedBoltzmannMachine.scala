package assignment4

import assignment3.DataBundle
import breeze.linalg._
import breeze.numerics._
import breeze.stats._
import com.typesafe.scalalogging.LazyLogging

/**
  * Created by Tom Lous on 08/01/2018.
  * Copyright © 2018 Datlinq B.V..
  */
case class RestrictedBoltzmannMachine(databundle:DataBundle) {


  // % first, train the rbm
//  def main(n_hid, lr_rbm, lr_classification, n_iterations) = {
//    global report_calls_to_sample_bernoulli
//      report_calls_to_sample_bernoulli = false;
//    global data_sets
//    if prod(size(data_sets)) ~= 1,
//    error('You must run a4_init before you do anything else.');
//    end
//    rbm_w = optimize([n_hid, 256], ...
//    @(rbm_w, data) cd1(rbm_w, data.inputs), ...  % discard labels
//    data_sets.training, ...
//    lr_rbm, ...
//    n_iterations);
//    % rbm_w is now a weight matrix of <n_hid> by <number of visible units, i.e. 256>
//      show_rbm(rbm_w);
//      input_to_hid = rbm_w;
//      % calculate the hidden layer representation of the labeled data
//      hidden_representation = logistic(input_to_hid * data_sets.training.inputs);
//      % train hid_to_class
//      data_2.inputs = hidden_representation;
//      data_2.targets = data_sets.training.targets;
//      hid_to_class = optimize([10, n_hid], @(model, data) classification_phi_gradient(model, data), data_2, lr_classification, n_iterations);
//      % report results
//      for data_details = reshape({'training', data_sets.training, 'validation', data_sets.validation, 'test', data_sets.test}, [2, 3]),
//    data_name = data_details{1};
//    data = data_details{2};
//    hid_input = input_to_hid * data.inputs; % size: <number of hidden units> by <number of data cases>
//      hid_output = logistic(hid_input); % size: <number of hidden units> by <number of data cases>
//        class_input = hid_to_class * hid_output; % size: <number of classes> by <number of data cases>
//          class_normalizer = log_sum_exp_over_rows(class_input); % log(sum(exp of class_input)) is what we subtract to get properly normalized log class probabilities. size: <1> by <number of data cases>
//            log_class_prob = class_input - repmat(class_normalizer, [size(class_input, 1), 1]); % log of probability of each class. size: <number of classes, i.e. 10> by <number of data cases>
//              error_rate = mean(double(argmax_over_rows(class_input) ~= argmax_over_rows(data.targets))); % scalar
//              loss = -mean(sum(log_class_prob .* data.targets, 1)); % scalar. select the right log class probability using that sum; then take the mean over all data cases.
//              fprintf('For the %s data, the classification cross-entropy loss is %f, and the classification error rate (i.e. the misclassification rate) is %f\n', data_name, loss, error_rate);
//              end
//              report_calls_to_sample_bernoulli = true;

//  }



//  def optimize(model_shape, gradient_function, training_data, learning_rate, n_iterations) = {

//    % This trains a model that's defined by a single matrix of weights.
//    % <model_shape> is the shape of the array of weights.
//      % <gradient_function> is a function that takes parameters <model> and <data> and returns the gradient (or approximate gradient in the case of CD-1) of the function that we're maximizing. Note the contrast with the loss function that we saw in PA3, which we were minimizing. The returned gradient is an array of the same shape as the provided <model> parameter.
//        % This uses mini-batches of size 100, momentum of 0.9, no weight decay, and no early stopping.
//        % This returns the matrix of weights of the trained model.
//        model = (a4_rand(model_shape, prod(model_shape)) * 2 - 1) * 0.1;
//        momentum_speed = zeros(model_shape);
//        mini_batch_size = 100;
//        start_of_next_mini_batch = 1;
//        for iteration_number = 1:n_iterations,
//        mini_batch = extract_mini_batch(training_data, start_of_next_mini_batch, mini_batch_size);
//        start_of_next_mini_batch = mod(start_of_next_mini_batch + mini_batch_size, size(training_data.inputs, 2));
//        gradient = gradient_function(model, mini_batch);
//        momentum_speed = 0.9 * momentum_speed + gradient;
//        model = model + momentum_speed * learning_rate;
//        end
//        end


//        }

}

object RestrictedBoltzmannMachine extends LazyLogging{
  /**
    * org: function ret = logistic(input)
    * @param input DenseMatrix
    * @return DenseMatrix
    */
  def logistic(input: DenseMatrix[Double]):DenseMatrix[Double] = 1.0 ./ (exp(-input) + 1.0)


  /**
    * describe_matrix
    * @param matrix
    */
  def describeMatrix(matrix: DenseMatrix[Double]):Unit = {
    logger.info(s"Describing a matrix of size ${matrix.rows} by ${matrix.cols}. The mean of the elements is ${mean(matrix)}. The sum of the elements is ${sum(matrix)}")
    //  fprintf('Describing a matrix of size %d by %d. The mean of the elements is %f. The sum of the elements is %f\n', size(matrix, 1), size(matrix, 2), mean(matrix(:)), sum(matrix(:)))
  }



}
