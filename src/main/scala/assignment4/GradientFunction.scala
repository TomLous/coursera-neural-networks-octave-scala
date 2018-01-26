package assignment4

import assignment3.DataBundle
import breeze.linalg._
import breeze.numerics._

/**
  * Created by Tom Lous on 09/01/2018.
  * Copyright © 2018 Datlinq B.V..
  */
trait GradientFunction {
  def run(rbmWeights: DenseMatrix[Double], visibleData: DataBundle, rbm:RestrictedBoltzmannMachine, reportCallsToSampleBernoulli: Boolean = false): DenseMatrix[Double]

}


/**
  * function ret = cd1(rbm_w, visible_data)
  * % <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
  * % <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
  * % The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.
  */
object CD1 extends GradientFunction {

  override def run(rbmWeights: DenseMatrix[Double], visibleData: DataBundle, rbm:RestrictedBoltzmannMachine, reportCallsToSampleBernoulli: Boolean = false): DenseMatrix[Double] =  runQ9(rbmWeights, visibleData, rbm, reportCallsToSampleBernoulli)


  def runQ7(rbmWeights: DenseMatrix[Double], visibleData: DataBundle, rbm:RestrictedBoltzmannMachine, reportCallsToSampleBernoulli: Boolean = false): DenseMatrix[Double] =  {
    // <solution Q7>
    val hiddenProbabilitiesSampled1 = rbm.sampleBernoulli(rbm.visibleStateToHiddenProbabilities(rbmWeights, visibleData.inputs), reportCallsToSampleBernoulli)
    val visibleProbabilitiesSampled2 = rbm.sampleBernoulli(rbm.hiddenStateToVisibleProbabilities(rbmWeights, hiddenProbabilitiesSampled1), reportCallsToSampleBernoulli)
    val hiddenProbabilitiesSampled2 = rbm.sampleBernoulli(rbm.visibleStateToHiddenProbabilities(rbmWeights, visibleProbabilitiesSampled2), reportCallsToSampleBernoulli)
    val configurationGoodnesssGradient1 = rbm.configurationGoodnesssGradient(visibleData.inputs, hiddenProbabilitiesSampled1)
    val configurationGoodnesssGradient2 = rbm.configurationGoodnesssGradient(visibleProbabilitiesSampled2, hiddenProbabilitiesSampled2)
    configurationGoodnesssGradient1 - configurationGoodnesssGradient2
    // </solution Q7>
  }

  def runQ8(rbmWeights: DenseMatrix[Double], visibleData: DataBundle, rbm:RestrictedBoltzmannMachine, reportCallsToSampleBernoulli: Boolean = false ): DenseMatrix[Double] =  {
    // <solution Q8>
    val hiddenProbabilitiesSampled1 = rbm.sampleBernoulli(rbm.visibleStateToHiddenProbabilities(rbmWeights, visibleData.inputs), reportCallsToSampleBernoulli)
    val visibleProbabilitiesSampled2 = rbm.sampleBernoulli(rbm.hiddenStateToVisibleProbabilities(rbmWeights, hiddenProbabilitiesSampled1), reportCallsToSampleBernoulli)
    val hiddenProbabilities2 = rbm.visibleStateToHiddenProbabilities(rbmWeights, visibleProbabilitiesSampled2)
    val configurationGoodnesssGradient1 = rbm.configurationGoodnesssGradient(visibleData.inputs, hiddenProbabilitiesSampled1)
    val configurationGoodnesssGradient2 = rbm.configurationGoodnesssGradient(visibleProbabilitiesSampled2, hiddenProbabilities2)
    configurationGoodnesssGradient1 - configurationGoodnesssGradient2
    // </solution Q8>
  }

  def runQ9(rbmWeights: DenseMatrix[Double], visibleData: DataBundle, rbm:RestrictedBoltzmannMachine, reportCallsToSampleBernoulli: Boolean = false ): DenseMatrix[Double] =  {
    val visibleProbabilitiesSampled1 = rbm.sampleBernoulli(visibleData.inputs, reportCallsToSampleBernoulli)
//    println(s"cd1 step 1: ${sum(visibleProbabilitiesSampled1)}")
//    println(s"cd1 step 1a1: ${sum(rbmWeights)}")
    val step1a = rbm.visibleStateToHiddenProbabilities(rbmWeights, visibleProbabilitiesSampled1)

//    println(s"cd1 step 1a2: ${sum(step1a)}")

    val hiddenProbabilitiesSampled1 = rbm.sampleBernoulli(step1a, reportCallsToSampleBernoulli)
//    println(s"cd1 step 2: ${sum(hiddenProbabilitiesSampled1)}")
    val visibleProbabilitiesSampled2 = rbm.sampleBernoulli(rbm.hiddenStateToVisibleProbabilities(rbmWeights, hiddenProbabilitiesSampled1), reportCallsToSampleBernoulli)
//    println(s"cd1 step 3: ${sum(visibleProbabilitiesSampled2)}")
    val hiddenProbabilities2 = rbm.visibleStateToHiddenProbabilities(rbmWeights, visibleProbabilitiesSampled2)
//    println(s"cd1 step 4: ${sum(hiddenProbabilities2)}")
//    val configurationGoodnesssGradient1 = rbm.configurationGoodnesssGradient(visibleData.inputs, hiddenProbabilitiesSampled1)
    val configurationGoodnesssGradient1 = rbm.configurationGoodnesssGradient(visibleProbabilitiesSampled1, hiddenProbabilitiesSampled1)
//    println(s"cd1 step 5: ${sum(configurationGoodnesssGradient1)}")
//    println(s"cd1 step 5a: ${sum(visibleData.inputs)}")
    val configurationGoodnesssGradient2 = rbm.configurationGoodnesssGradient(visibleProbabilitiesSampled2, hiddenProbabilities2)
//    println(s"cd1 step 6: ${sum(configurationGoodnesssGradient2)}")
    configurationGoodnesssGradient1 - configurationGoodnesssGradient2

  }
}

/**
  * This is about a very simple model: there's an input layer, and a softmax output layer. There are no hidden layers, and no biases.
  * This returns the gradient of phi (a.k.a. negative the loss) for the <input_to_class> matrix.
  * <input_to_class> is a matrix of size <number of classes> by <number of input units>.
  * <data> has fields .inputs (matrix of size <number of input units> by <number of data cases>) and .targets (matrix of size <number of classes> by <number of data cases>).
  * first: forward pass
  */
object ClassificationPhiGradient extends GradientFunction {

  override def run(inputToClassification: DenseMatrix[Double], data: DataBundle, rbm:RestrictedBoltzmannMachine, reportCallsToSampleBernoulli: Boolean = false ): DenseMatrix[Double] = {
    // input to the components of the softmax. size: <number of classes> by <number of data cases>
    val classificationInput: DenseMatrix[Double] = inputToClassification * data.inputs // class_input = input_to_class * data.inputs

    // log(sum(exp)) is what we subtract to get normalized log class probabilities. size: <1> by <number of data cases>
    val classificationNormalizer = rbm.logSumExpOverRows(classificationInput) //class_normalizer = log_sum_exp_over_rows(class_input);

    val tiledNormalizer = tile(classificationNormalizer, 1, classificationInput.rows).t

    //log of probability of each class. size: <number of classes, i.e. 10> by <number of data cases>
    val logClassificationProbability = classificationInput - tiledNormalizer // log_class_prob = class_input - repmat(class_normalizer, [size(class_input, 1), 1]);

    //   probability of each class. Each column (i.e. each case) sums to 1. size: <number of classes> by <number of data cases>
    val classificationProbability = exp(logClassificationProbability) //exp(log_class_prob);


    // size: <number of classes> by <number of data cases>
    val dLossByDClassificationInput: DenseMatrix[Double] = -(data.targets - classificationProbability) / data.inputs.cols.toDouble // d_loss_by_d_class_input = -(data.targets - class_prob) ./ size(data.inputs, 2);

    // size: <number of classes> by <number of input units>
    val dLossByDInputToClassification = dLossByDClassificationInput * data.inputs.t //  d_loss_by_d_input_to_class = d_loss_by_d_class_input * data.inputs.'; %

    val dPhiByDInputToClassification = -dLossByDInputToClassification //  d_phi_by_d_input_to_class = -d_loss_by_d_input_to_class;

    dPhiByDInputToClassification

  }

}