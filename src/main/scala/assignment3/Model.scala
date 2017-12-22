package assignment3

import breeze.linalg._
import breeze.numerics._
import breeze.stats._




/**
  * Created by Tom Lous on 10/10/2017.
  * Copyright © 2017 Datlinq B.V..
  */
case class Model(numberHiddenUnits: Int, inputToHidden: DenseMatrix[Double], hiddenToClassification:DenseMatrix[Double]) {


  /**
    * This function takes a model (or gradient in model form), and turns it into one long vector. See also theta_to_model.
    * org: function ret = model_to_theta(model)
    */
  lazy val theta:Theta = {
    val input_to_hid_transpose = inputToHidden.t
    val hid_to_class_transpose = hiddenToClassification.t

    Theta(DenseVector.vertcat(input_to_hid_transpose.toDenseVector, hid_to_class_transpose.toDenseVector), Model.NUM_INPUT_UNITS, Model.NUM_OUTPUT_UNITS)
  }


  /**
    * This returns the fraction of data cases that is incorrectly classified by the model.
    * function ret = classification_performance(model, data)
    * @param data DataBundle to test on
    * @return Double fraction of incorrectly classifications
    */
  def classificationPerformance(data: DataBundle):Double = {

    val (_,_, classificationInput) = forwardPass(data)

    // % choices is integer: the chosen class
    val choices:DenseVector[Int] = classificationInput(::,*).map(dv => argmax(dv)).t   // [dump, choices] = max(class_input);

    // % targets is integer: the target class
    val targets:DenseVector[Int] = data.targets(::,*).map(dv => argmax(dv)).t // [dump, targets] = max(data.targets);

    // Matlab has a nice method ~= to determine inequality. I just subtract the two vectors and any non-zero gets mapped to 1
    val ret = mean((choices-targets).map(x => (math.abs(x) min 1).toDouble)) // ret = mean(double(choices ~= targets));

    ret
  }

  /**
    * The returned object is supposed to be exactly like parameter <model>, i.e. it has fields ret.input_to_hid and ret.hid_to_class. However, the contents of those matrices are gradients (d loss by d model parameter), instead of model parameters.
    * org: function ret = d_loss_by_d_model(model, data, wd_coefficient)
    * - model.input_to_hid is a matrix of size <number of hidden units> by <number of inputs i.e. 256
    * - model.hid_to_class is a matrix of size <number of classes i.e. 10> by <number of hidden units>
    * - data.inputs is a matrix of size <number of inputs i.e. 256> by <number of data cases>. Each column describes a different data case.
    * - data.targets is a matrix of size <number of classes i.e. 10> by <number of data cases>. Each column describes a different data case. It contains a one-of-N encoding of the class, i.e. one element in every column is 1 and the others are 0.
    * @param data DataBundle of cases
    * @param weightDecayCoefficient Coeff
    * @return Model (new Model based on current)
    */
  def dLossBydModel(data: DataBundle, weightDecayCoefficient: Double):Model = {
    // % This is the only function that you're expected to change. Right now, it just returns a lot of zeros, which is obviously not the correct output. Your job is to replace that by a correct computation.

//    val newInputToHidden: DenseMatrix[Double] = inputToHidden * 0.0
//    val newHiddenToClassification:DenseMatrix[Double] = hiddenToClassification * 0.0


    // If you want to find solution yourself, uncomment lines above and delete everything between <solution> bracktes
    // <solution>

    val m1 = numberHiddenUnits
    val m2 = sum(this.inputToHidden)
    val m3 = sum(this.hiddenToClassification)
    val a0 = sum(data.inputs)
    val a1 = sum(data.targets)

    // Forward pass
    val (hi, hiddenOutput,classificationInput) = forwardPass(data)

    val a1b = sum(hi)
    val a2 = sum(hiddenOutput)
    val a3 = sum(classificationInput)

    // Softmax
    val (lcp,classificationProbability) = softmax(classificationInput)


    val a4 = sum(lcp)
    val a5 = sum(classificationProbability)

    val m = data.inputs.cols.toDouble

    val a6 = m

    // Backward Pass

    // New newHiddenToClassification
    val delta = classificationProbability - data.targets //class_prob-data.targets

    val a7 = sum(delta)

    val htcGradient1 = delta * hiddenOutput.t

    val perCaseHTCGradient = htcGradient1 / m

    val L2a = weightDecayCoefficient * hiddenToClassification

    val newHiddenToClassification = perCaseHTCGradient + L2a

    val a8 = sum(newHiddenToClassification)


    // newInputToHidden

    val htcGradient2 = hiddenToClassification.t * delta

    val errorFactor =  hiddenOutput *:* (1.0 - hiddenOutput)

    val errorDerivative = htcGradient2 *:* errorFactor

    val a9 = sum(errorDerivative)

    val ithClean =  data.inputs * errorDerivative.t

    val L2b = weightDecayCoefficient * inputToHidden





    val newInputToHidden = ithClean / m + L2b

    val a10 = sum(newInputToHidden)

    // </solution>

    Model(numberHiddenUnits, newInputToHidden, newHiddenToClassification)
  }


  /**
    * This computes log(sum(exp(a), 1)) in a numerically stable way
    * @param a DenseMatrix
    * @return ret DenseVector[Double]
    */
  def logSumExpOverRows(a: DenseMatrix[Double]): DenseVector[Double] ={
      val maxs_small = max(a, Axis._0) // maxs_small = max(a, [], 1);
      val maxs_big = tile(maxs_small, 1, a.rows)    // maxs_big = repmat(maxs_small, [size(a, 1), 1]);
      val ret = log(sum(exp(a - maxs_big), Axis._0)) + maxs_small

      ret.t
  }

  /**
    * Calculate the loss
    * - model.input_to_hid is a matrix of size <number of hidden units> by <number of inputs i.e. 256>. It contains the weights from the input units to the hidden units.
    * - model.hid_to_class is a matrix of size <number of classes i.e. 10> by <number of hidden units>. It contains the weights from the hidden units to the softmax units.
    * - data.inputs is a matrix of size <number of inputs i.e. 256> by <number of data cases>. Each column describes a different data case.
    * - data.targets is a matrix of size <number of classes i.e. 10> by <number of data cases>. Each column describes a different data case. It contains a one-of-N encoding of the class, i.e. one element in every column is 1 and the others are 0.
    *  org: function ret = loss(model, data, wd_coefficient)
    * @param data DataBundle
    * @param weightDecayCoefficient Double
    * @return Double loss
    */
  def loss(data: DataBundle, weightDecayCoefficient: Double):Double = {
    // Before we can calculate the loss, we need to calculate a variety of intermediate values, like the state of the hidden units.


    val (_, _, classificationInput) = forwardPass(data)


    val(logClassificationProbability,_) = softmax(classificationInput)

    //select the right log class probability using that sum; then take the mean over all data cases.
    val classificationLoss:Double = -mean(sum(logClassificationProbability *:* data.targets, Axis._0)) // -mean(sum(log_class_prob .* data.targets, 1));

    // weight decay loss. very straightforward: E = 1/2 * wd_coeffecient * theta^2
    val weightDecayLoss:Double = (sum(pow(this.theta.thetaVector,2))  / 2) * weightDecayCoefficient  // wd_loss = sum(model_to_theta(model).^2)/2*wd_coefficient; %


    classificationLoss + weightDecayLoss
  }

  private def forwardPass(data: DataBundle):(DenseMatrix[Double],DenseMatrix[Double],DenseMatrix[Double]) = {

    val ith2 = inputToHidden.t.toDenseMatrix
    val m1 = numberHiddenUnits
    val m2 = sum(ith2)
    val m3 = sum(hiddenToClassification)

    val m4 = ith2.rows
    val m5 = ith2.cols

    val ba = data.inputs.rows
    val bb = data.inputs.cols

    val b0 = sum(data.inputs)
    // input to the hidden units, i.e. before the logistic. size: <number of hidden units> by <number of data cases>
    val hiddenInput:DenseMatrix[Double] = ith2 * data.inputs  // hid_input = model.input_to_hid * data.inputs;



    val ba2 = hiddenInput.rows
    val bb2 = hiddenInput.cols

    val b1 = sum(hiddenInput)

    // output of the hidden units, i.e. after the logistic. size: <number of hidden units> by <number of data cases>
    val hiddenOutput:DenseMatrix[Double] = Model.logistic(hiddenInput)  // logistic(hid_input);

    val b2 = sum(hiddenOutput)

    // input to the components of the softmax. size: <number of classes, i.e. 10> by <number of data cases>
    val classificationInput:DenseMatrix[Double] = hiddenToClassification * hiddenOutput  // class_input = model.hid_to_class * hid_output;

    val b3 = sum(classificationInput)

    (hiddenInput, hiddenOutput,classificationInput)
  }

  private def softmax(classificationInput:DenseMatrix[Double]):(DenseMatrix[Double],DenseMatrix[Double]) = {
    // The following four lines of code implement the softmax.
    // However, it's written differently from what the lectures say.
    //  In the lectures, a softmax is described using an exponential divided by a sum of exponentials.
    //  What we do here is exactly equivalent (you can check the math or just check it in practice), but this is more numerically stable.
    // "Numerically stable" means that this way, there will never be really big numbers involved.
    //  The exponential in the lectures can lead to really big numbers, which are fine in mathematical equations, but can lead to all sorts of problems in Octave.
    //  Octave isn't well prepared to deal with really large numbers, like the number 10 to the power 1000. Computations with such numbers get unstable, so we avoid them.

    // log(sum(exp of class_input)) is what we subtract to get properly normalized log class probabilities. size: <1> by <number of data cases>
    val classificationNormalizer:DenseVector[Double] = logSumExpOverRows(classificationInput)  // class_normalizer = log_sum_exp_over_rows(class_input);


    val tiledNormalizer = tile(classificationNormalizer, 1, classificationInput.rows).t
    // log of probability of each class. size: <number of classes, i.e. 10> by <number of data cases>
    val logClassificationProbability:DenseMatrix[Double] = classificationInput - tiledNormalizer //class_input - repmat(class_normalizer, [size(class_input, 1), 1]);

    // probability of each class. Each column (i.e. each case) sums to 1. size: <number of classes, i.e. 10> by <number of data cases>
    val classificationProbability:DenseMatrix[Double] = exp(logClassificationProbability) // class_prob = exp(log_class_prob);

    (logClassificationProbability, classificationProbability)
  }


}

object Model{

  val NUM_INPUT_UNITS:Int = 256
  val NUM_OUTPUT_UNITS:Int = 10 // digits

  /**
    * org: function ret = initial_model(n_hid)
    * @param numberHiddenUnits amount of hidden units
    * @return Model
    */
  def apply(numberHiddenUnits: Int): Model = {
    // n_params = (256+10) * n_hid;
    val numberOfParams = (NUM_INPUT_UNITS+NUM_OUTPUT_UNITS) * numberHiddenUnits

    // as_row_vector = cos(0:(n_params-1));
    val asRowVector = cos(DenseVector[Double](
      (0 until numberOfParams)
        .map(_.toDouble):_*)
    )

    // ret = theta_to_model(as_row_vector(:) * 0.1); % We don't use random initialization, for this assignment. This way, everybody will get the same results.
    Theta(asRowVector * 0.1, NUM_INPUT_UNITS, NUM_OUTPUT_UNITS).model
  }


  /**
    * org: function ret = logistic(input)
    * @param input DenseMatrix
    * @return DenseMatrix
    */
  def logistic(input: DenseMatrix[Double]):DenseMatrix[Double] = 1.0 ./ (exp(-input) + 1.0)


}
