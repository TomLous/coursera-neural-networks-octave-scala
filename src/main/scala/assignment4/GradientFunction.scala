package assignment4

import assignment3.DataBundle
import breeze.linalg._
import breeze.numerics._

/**
  * Created by Tom Lous on 09/01/2018.
  * Copyright © 2018 Datlinq B.V..
  */
trait GradientFunction {
  def run(rbmWeights: DenseMatrix[Double], visibleData: DataBundle): DenseMatrix[Double]

}


object CD1 extends GradientFunction {
  override def run(rbmWeights: DenseMatrix[Double], visibleData: DataBundle): DenseMatrix[Double] = ???
}

/**
  * This is about a very simple model: there's an input layer, and a softmax output layer. There are no hidden layers, and no biases.
  * This returns the gradient of phi (a.k.a. negative the loss) for the <input_to_class> matrix.
  * <input_to_class> is a matrix of size <number of classes> by <number of input units>.
  * <data> has fields .inputs (matrix of size <number of input units> by <number of data cases>) and .targets (matrix of size <number of classes> by <number of data cases>).
  * first: forward pass
  */
object ClassificationPhiGradient extends GradientFunction {

  override def run(inputToClassification: DenseMatrix[Double], data: DataBundle): DenseMatrix[Double] = {
    // input to the components of the softmax. size: <number of classes> by <number of data cases>
    val classificationInput: DenseMatrix[Double] = inputToClassification * data.inputs // class_input = input_to_class * data.inputs

    // log(sum(exp)) is what we subtract to get normalized log class probabilities. size: <1> by <number of data cases>
    val classificationNormalizer = RestrictedBoltzmannMachine.logSumExpOverRows(classificationInput) //class_normalizer = log_sum_exp_over_rows(class_input);

    val tiledNormalizer = tile(classificationNormalizer, 1, classificationInput.rows).t

    //log of probability of each class. size: <number of classes, i.e. 10> by <number of data cases>
    val logClassificationProbability = classificationInput - tiledNormalizer // log_class_prob = class_input - repmat(class_normalizer, [size(class_input, 1), 1]);

    //   probability of each class. Each column (i.e. each case) sums to 1. size: <number of classes> by <number of data cases>
    val classificationProbability = exp(logClassificationProbability) //exp(log_class_prob);


    // size: <number of classes> by <number of data cases>
    val dLossByDClassificationInput: DenseMatrix[Double] = -(data.targets - classificationProbability) / data.inputs.cols.toDouble // d_loss_by_d_class_input = -(data.targets - class_prob) ./ size(data.inputs, 2);

    // size: <number of classes> by <number of input units>
    val dLossByDInputToClassification = dLossByDClassificationInput *:* data.inputs.t //  d_loss_by_d_input_to_class = d_loss_by_d_class_input * data.inputs.'; %

    val dPhiByDInputToClassification = -dLossByDInputToClassification //  d_phi_by_d_input_to_class = -d_loss_by_d_input_to_class;

    dPhiByDInputToClassification

  }

}