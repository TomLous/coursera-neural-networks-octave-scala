package assignment4

import assignment3.DataBundle
import assignment4.Utils._
import breeze.linalg._
import breeze.numerics._
import breeze.stats._
import com.sksamuel.scrimage.canvas.{Context, DrawableString, _}
import com.sksamuel.scrimage.nio.PngWriter
import com.sksamuel.scrimage.{Color, Grayscale, Image, Pixel}
import com.typesafe.scalalogging.LazyLogging

import scala.util.{Failure, Success, Try}

/**
  * Created by Tom Lous on 08/01/2018.
  * Copyright © 2018 Datlinq B.V..
  */
case class RestrictedBoltzmannMachine(trainingData: DataBundle, validationData: DataBundle, testData: DataBundle, randomDataSource:RandomDataSource)  extends LazyLogging {

  val testRbmWeights = randomDataSource.rand(MatrixSize(100, 256), 0.0) * 2.0 - 1.0 //    test_rbm_w = a4_rand([100, 256], 0) * 2 - 1;
  val smallTestRbmWeights = randomDataSource.rand(MatrixSize(100, 256), 0.0) * 2.0 - 1.0 //small_test_rbm_w = a4_rand([10, 256], 0) * 2 - 1;

  val temp = trainingData.batch(0, 1) //    temp = extract_mini_batch(data_sets.training, 1, 1);
  val data1Case = sampleBernoulli(temp.inputs) //    data_1_case = sample_bernoulli(temp.inputs);

  val temp2 = trainingData.batch(99, 10) //    temp = extract_mini_batch(data_sets.training, 100, 10);
  val data10Cases = sampleBernoulli(temp2.inputs) //    data_10_cases = sample_bernoulli(temp.inputs);

  val temp3 = trainingData.batch(199, 37) //   temp = extract_mini_batch(data_sets.training, 200, 37);
  val data37Cases = sampleBernoulli(temp3.inputs) //     data_37_cases = sample_bernoulli(temp.inputs);


  val testHiddenState1Case = sampleBernoulli(randomDataSource.rand(MatrixSize(100,1),0.0)) //    test_hidden_state_1_case = sample_bernoulli(a4_rand([100, 1], 0));
  val testHiddenState10Cases = sampleBernoulli(randomDataSource.rand(MatrixSize(100,10),1.0))//    test_hidden_state_10_cases = sample_bernoulli(a4_rand([100, 10], 1));
  val testHiddenState37Cases = sampleBernoulli(randomDataSource.rand(MatrixSize(100,37),2.0))//    test_hidden_state_37_cases = sample_bernoulli(a4_rand([100, 37], 2));



  def Q3() = {
    val (dimM,meanM, sumM) = describeMatrix("Q3. data1Case: ", visibleStateToHiddenProbabilities(testRbmWeights,data1Case)) //describe_matrix(visible_state_to_hidden_probabilities(test_rbm_w, data_1_case))

    assertDouble(meanM, 0.447562)
    assertDouble(sumM, 44.756160)
    assertDimensions(dimM, MatrixSize(100, 1))

    val (dimM2,meanM2, sumM2) = describeMatrix("Q3. data10Cases: ", visibleStateToHiddenProbabilities(testRbmWeights,data10Cases)) //describe_matrix(visible_state_to_hidden_probabilities(test_rbm_w, data_10_cases))

    assertDouble(meanM2, 0.459927)
    assertDouble(sumM2, 459.927012)
    assertDimensions(dimM2, MatrixSize(100, 10))

    describeMatrix("Q3. data37Cases: ", visibleStateToHiddenProbabilities(testRbmWeights,data37Cases)) // describe_matrix(visible_state_to_hidden_probabilities(test_rbm_w, data_37_cases))
  }


  def Q4() = {
    val (dimM,meanM, sumM) = describeMatrix("Q4. testHiddenState1Case: ", hiddenStateToVisibleProbabilities(testRbmWeights,testHiddenState1Case)) // describe_matrix(hidden_state_to_visible_probabilities(test_rbm_w, test_hidden_state_1_case))

    assertDouble(meanM, 0.474996)
    assertDouble(sumM, 121.598898)
    assertDimensions(dimM, MatrixSize(256, 1))

    val (dimM2,meanM2, sumM2) = describeMatrix("Q4. testHiddenState10Cases: ", hiddenStateToVisibleProbabilities(testRbmWeights,testHiddenState10Cases)) //describe_matrix(hidden_state_to_visible_probabilities(test_rbm_w, test_hidden_state_10_cases))

    assertDouble(meanM2, 0.469464)
    assertDouble(sumM2, 1201.828527)
    assertDimensions(dimM2, MatrixSize(256, 10))

    describeMatrix("Q4. testHiddenState37Cases: ", hiddenStateToVisibleProbabilities(testRbmWeights,testHiddenState37Cases)) // describe_matrix(hidden_state_to_visible_probabilities(test_rbm_w, test_hidden_state_37_cases))
  }

  def Q5() = {

    val mean1 = configurationGoodnesss(testRbmWeights, data1Case, testHiddenState1Case) // configuration_goodness(test_rbm_w, data_1_case, test_hidden_state_1_case)
    assertDouble(mean1, 13.5399, 0.01 )

    val mean2 = configurationGoodnesss(testRbmWeights, data10Cases, testHiddenState10Cases) // configuration_goodness(test_rbm_w, data_10_cases, test_hidden_state_10_cases)
    assertDouble(mean2, -32.9614, 0.01 )


    val mean3 = configurationGoodnesss(testRbmWeights, data37Cases, testHiddenState37Cases) // configuration_goodness(test_rbm_w, data_37_cases, test_hidden_state_37_cases)
    logger.info(s"Q5. configurationGoodnesss(testRbmWeights, data37Cases, testHiddenState37Cases) = $mean3")
  }

  def Q6() = {
    val (dimM,meanM, sumM) = describeMatrix("Q6. data1Case & testHiddenState1Case: ", configurationGoodnesssGradient(data1Case,testHiddenState1Case)) // describe_matrix(configuration_goodness_gradient(data_1_case, test_hidden_state_1_case))

    assertDouble(meanM, 0.159922)
    assertDouble(sumM, 4094.000000)
    assertDimensions(dimM, MatrixSize(100, 256))

    val (dimM2,meanM2, sumM2) = describeMatrix("Q6. data10Cases & testHiddenState10Cases: ", configurationGoodnesssGradient(data10Cases,testHiddenState10Cases)) // describe_matrix(configuration_goodness_gradient(data_10_cases, test_hidden_state_10_cases))

    assertDouble(meanM2, 0.116770)
    assertDouble(sumM2, 2989.300000)
    assertDimensions(dimM2, MatrixSize(100, 256))

    describeMatrix("Q6. data37Cases & testHiddenState37Cases: ", configurationGoodnesssGradient(data37Cases,testHiddenState37Cases)) // describe_matrix(configuration_goodness_gradient(data_37_cases, test_hidden_state_37_cases))

  }

  def Q7() = {
    val (dimM,meanM, sumM) = describeMatrix("Q7. CD1 & data1Case: ", CD1.runQ7(testRbmWeights, DataBundle(data1Case, null), this)) // describe_matrix(cd1(test_rbm_w, data_1_case))

    assertDouble(meanM, -0.160742)
    assertDouble(sumM, -4115.000000)
    assertDimensions(dimM, MatrixSize(100, 256))

    val (dimM2,meanM2, sumM2) = describeMatrix("Q7. CD1 & data10Cases: ", CD1.runQ7(testRbmWeights, DataBundle(data10Cases, null), this)) // describe_matrix(configuration_goodness_gradient(data_10_cases, test_hidden_state_10_cases))

    assertDouble(meanM2, -0.185137)
    assertDouble(sumM2, -4739.500000)
    assertDimensions(dimM2, MatrixSize(100, 256))

    describeMatrix("Q7. CD1 & data37Cases : ",  CD1.runQ7(testRbmWeights, DataBundle(data37Cases, null), this)) // describe_matrix(configuration_goodness_gradient(data_37_cases, test_hidden_state_37_cases))

  }

  def Q8() = {
    val (dimM,meanM, sumM) = describeMatrix("Q8. CD1 & data1Case: ", CD1.runQ8(testRbmWeights, DataBundle(data1Case, null), this)) // describe_matrix(cd1(test_rbm_w, data_1_case))

    assertDouble(meanM, -0.164335)
    assertDouble(sumM, -4206.981332)
    assertDimensions(dimM, MatrixSize(100, 256))

    val (dimM2,meanM2, sumM2) = describeMatrix("Q8. CD1 & data10Cases: ", CD1.runQ8(testRbmWeights, DataBundle(data10Cases, null), this)) // describe_matrix(configuration_goodness_gradient(data_10_cases, test_hidden_state_10_cases))

    assertDouble(meanM2, -0.185591)
    assertDouble(sumM2, -4751.142054)
    assertDimensions(dimM2, MatrixSize(100, 256))

    describeMatrix("Q8. CD1 & data37Cases : ",  CD1.runQ8(testRbmWeights, DataBundle(data37Cases, null), this)) // describe_matrix(configuration_goodness_gradient(data_37_cases, test_hidden_state_37_cases))

  }


  /**
    * function a4_main(n_hid, lr_rbm, lr_classification, n_iterations)
    * % first, train the rbm
    * @param numberHidden n_hid
    * @param learningRateRBM lr_rbm
    * @param learningRateClassification lr_classification
    * @param numberOfIterations n_iterations
    */
  def main(id: String, numberHidden:Int, learningRateRBM:Double, learningRateClassification:Double, numberOfIterations:Int) = {
    val rbmWeights = optimize(MatrixSize(numberHidden, 256),CD1, trainingData, learningRateRBM, numberOfIterations) //    rbm_w = optimize([n_hid, 256], @(rbm_w, data) cd1(rbm_w, data.inputs), data_sets.training, lr_rbm,   n_iterations);

    // rbm_w is now a weight matrix of <n_hid> by <number of visible units, i.e. 256>
    show(id, rbmWeights) //  show_rbm(rbm_w);

    val inputToHidden = rbmWeights //      input_to_hid = rbm_w;

    // calculate the hidden layer representation of the labeled data
    val hiddenRepresentation = logistic(inputToHidden * trainingData.inputs) //      hidden_representation = logistic(input_to_hid * data_sets.training.inputs);

    // train hid_to_class
    val data2 = DataBundle(hiddenRepresentation, trainingData.targets) //      data_2.inputs = hidden_representation; data_2.targets = data_sets.training.targets;
    val hiddenToClassification = optimize(MatrixSize(10, numberHidden), ClassificationPhiGradient, data2, learningRateClassification, numberOfIterations) //      hid_to_class = optimize([10, n_hid], @(model, data) classification_phi_gradient(model, data), data_2, lr_classification, n_iterations);

    // report results
    val dataDetails = Map( //      for data_details = reshape({'training', data_sets.training, 'validation', data_sets.validation, 'test', data_sets.test}, [2, 3]),
      "training" -> trainingData,
      "validation" -> validationData,
      "test" -> testData
    )


    dataDetails.foreach{
      case (dataName, data) => //    data_name = data_details{1}; data = data_details{2};

        // size: <number of hidden units> by <number of data cases>
        val hiddenInput = inputToHidden * data.inputs //    hid_input = input_to_hid * data.inputs;

        // size: <number of hidden units> by <number of data cases>
        val hiddenOutput = logistic(hiddenInput) //      hid_output = logistic(hid_input); %

        // size: <number of classes> by <number of data cases>
        val classificationInput = hiddenToClassification * hiddenOutput // class_input = hid_to_class * hid_output;

        // log(sum(exp of class_input)) is what we subtract to get properly normalized log class probabilities. size: <1> by <number of data cases>
        val classificationNormalizer = logSumExpOverRows(classificationInput) //class_normalizer = log_sum_exp_over_rows(class_input);


        val tiledNormalizer = tile(classificationNormalizer, 1, classificationInput.rows).t

        //log of probability of each class. size: <number of classes, i.e. 10> by <number of data cases>
        val logClassificationProbability = classificationInput - tiledNormalizer // log_class_prob = class_input - repmat(class_normalizer, [size(class_input, 1), 1]);

        // Matlab has a nice method ~= to determine inequality. I just subtract the two vectors and any non-zero gets mapped to 1
        val errorRate:Double = mean((argmaxOverRows(classificationInput) - argmaxOverRows(data.targets)).map(x => (math.abs(x) min 1).toDouble))   // error_rate = mean(double(argmax_over_rows(class_input) ~= argmax_over_rows(data.targets))); % scalar


        //  scalar. select the right log class probability using that sum; then take the mean over all data cases.
        val loss:Double = -mean(sum(logClassificationProbability *:* data.targets, Axis._0)) // loss = -mean(sum(log_class_prob .* data.targets, 1))

        logger.info(s"$id For the $dataName data, the classification cross-entropy loss is $loss, and the classification error rate (i.e. the misclassification rate) is $errorRate")

    }

  }


  /**
    * function model = optimize(model_shape, gradient_function, training_data, learning_rate, n_iterations)
    * This trains a model that's defined by a single matrix of weights.
    * @param modelShape is the shape of the array of weights.
    * @param gradientFunction is a function that takes parameters <model> and <data> and returns the gradient (or approximate gradient in the case of CD-1) of the function that we're maximizing. Note the contrast with the loss function that we saw in PA3, which we were minimizing. The returned gradient is an array of the same shape as the provided <model> parameter.
    * @param learningRate  learning_rate
    * @param numberOfIterations n_iterations
    */
  def optimize(modelShape:MatrixSize, gradientFunction: GradientFunction, data:DataBundle, learningRate:Double, numberOfIterations:Int):DenseMatrix[Double] = {
    val model = (randomDataSource.rand(modelShape, modelShape.prod) * 2.0 - 1.0) * 0.1 //        model = (a4_rand(model_shape, prod(model_shape)) * 2 - 1) * 0.1;
    val momentumSpeed = DenseMatrix.zeros[Double](modelShape.rows, modelShape.cols) //        momentum_speed = zeros(model_shape);
    val miniBatchSize = 100; //mini_batch_size = 100;

    val (_,_,rbmWeights) =  (0 until numberOfIterations).foldLeft(0, momentumSpeed, model){ //        for iteration_number = 1:n_iterations,
      case ((startOfNextMiniBatch, currentMomentumSpeed, currentModel), iterationNumber) =>
        val miniBatch = data.batch(startOfNextMiniBatch, miniBatchSize) // mini_batch = extract_mini_batch(training_data, start_of_next_mini_batch, mini_batch_size);

        val nextStartOfNextMiniBatch = (startOfNextMiniBatch + miniBatchSize) % data.inputs.cols  //        start_of_next_mini_batch = mod(start_of_next_mini_batch + mini_batch_size, size(training_data.inputs, 2));

        val gradient = gradientFunction.run(currentModel, miniBatch, this) // gradient = gradient_function(model, mini_batch);

        val newMomentumSpeed = 0.9 * currentMomentumSpeed  //  momentum_speed = 0.9 * momentum_speed + gradient;

        val newModel = currentModel + momentumSpeed * learningRate // model = model + momentum_speed * learning_rate;

        (nextStartOfNextMiniBatch, newMomentumSpeed, newModel)
    }

    rbmWeights
  }


  /**
    * function show_rbm(rbm_w) Generates and saves visual representation f hidden units to file
    * @param id
    * @param rbmWeights
    */
  def show(id: String, rbmWeights:DenseMatrix[Double]): Unit ={
    Try {
      val numberHidden = rbmWeights.rows //    n_hid = size(rbm_w, 1);
      val numberRows = math.ceil(math.sqrt(numberHidden)).toInt //    n_rows = ceil(sqrt(n_hid));
      val blankLines = 4 //    blank_lines = 4;
      val squareSize = 16

      val distance = squareSize + blankLines //    distance = 16 + blank_lines;
      val toShow = DenseMatrix.zeros[Double](numberRows * distance + blankLines, numberRows * distance + blankLines) //    to_show = zeros([n_rows * distance + blank_lines, n_rows * distance + blank_lines]);
      (0 until numberHidden).foreach(i => { //    for i = 0:n_hid-1,
        val rowI = math.floor(i / numberRows).toInt // row_i = floor(i / n_rows);
        val colI = i % numberRows //     col_i = mod(i, n_rows);
        val pixels: DenseMatrix[Double] = rbmWeights(i, ::).t.toDenseMatrix.reshape(squareSize, squareSize).t ////    pixels = reshape(rbm_w(i+1, :), [16, 16]).';

        val rowBase = rowI * distance + blankLines //    row_base = row_i*distance + blank_lines;
        val colBase = colI * distance + blankLines //    col_base = col_i*distance + blank_lines;
        toShow(rowBase until rowBase + squareSize, colBase until colBase + squareSize) := pixels // //    to_show(row_base+1:row_base+16, col_base+1:col_base+16) = pixels;
      })

      val imageSize = toShow.cols
      val extreme = max(abs(toShow)) //  extreme = max(abs(to_show(:)));

      def grayScale(num: Double): Pixel = {
        val rangeFactor = (num + extreme) / (extreme * 2)

        Grayscale((rangeFactor * 255).toInt).toPixel
      }

      val pixels: Array[Pixel] = toShow.t.toArray.map(grayScale)

      implicit val writer = PngWriter.NoCompression
      import Canvas._

      val image = Image(imageSize, imageSize, pixels)
        .padTop(20, Color.White)
        .draw(DrawableString("hidden units of the RBM", imageSize/2 - 60, 14, context = Context.Aliased, font = new java.awt.Font("default", 0, 10)))

      val path = "src/main/resources/assignment4/image"
      val name = "hidden-units-" + id.replaceAll("""\W+""", " ").trim.replaceAll("""\s+""", "-")
      val file = new java.io.File(s"$path/$name.png")
      image.output(file) // imshow(to_show, [-extreme, extreme]);
    } match {
      case Failure(_) =>
        logger.warn(s"$id Failed to display the RBM. No big deal (you do not need the display to finish the assignment), but you are missing out on an interesting picture.")
      case Success(f) => logger.info(s"$id Saved image $f")
    }
  }


  /**
    * binary = sample_bernoulli(probabilities)
    * @param probabilities
    * @param reportCallsToSampleBernoulli
    * @return
    */
  def sampleBernoulli(probabilities:DenseMatrix[Double], reportCallsToSampleBernoulli: Boolean = false):DenseMatrix[Double] = {

    if(reportCallsToSampleBernoulli) { //if report_calls_to_sample_bernoulli,
      logger.info(s"sample_bernoulli() was called with a matrix of size ${probabilities.rows} by ${probabilities.cols}. ") //fprintf('sample_bernoulli() was called with a matrix of size %d by %d. ', size(probabilities, 1), size(probabilities, 2));
    }

    val seed:Double = sum(probabilities) //seed = sum(probabilities(:));

    val rand = randomDataSource.rand(MatrixSize(probabilities), seed)

    (probabilities >:> rand).map(b => if (b) 1.0 else 0.0) // binary = +(probabilities > a4_rand(size(probabilities), seed)); % the "+" is to avoid the "logical" data type, which just confuses things.
  }


  /**
    * describe_matrix
    * @param matrix
    */
  def describeMatrix(id: String, matrix: DenseMatrix[Double]):(MatrixSize, Double, Double) = {
    val meanM = mean(matrix)
    val sumM = sum(matrix)


    logger.info(s"$id Describing a matrix of size ${matrix.rows} by ${matrix.cols}. The mean of the elements is $meanM. The sum of the elements is $sumM")
    //  fprintf('Describing a matrix of size %d by %d. The mean of the elements is %f. The sum of the elements is %f\n', size(matrix, 1), size(matrix, 2), mean(matrix(:)), sum(matrix(:)))

    (MatrixSize(matrix), meanM, sumM)
  }


  /**
    * org: function ret = logistic(input)
    * @param input DenseMatrix
    * @return DenseMatrix
    */
  def logistic(input: DenseMatrix[Double]):DenseMatrix[Double] = 1.0 ./ (exp(-input) + 1.0)




  /**
    * This computes log(sum(exp(a), 1)) in a numerically stable way
    * @param a DenseMatrix
    * @return ret DenseVector[Double]
    */
  def logSumExpOverRows(a: DenseMatrix[Double]): DenseVector[Double] ={
    val maxsSmall = max(a, Axis._0) // maxs_small = max(a, [], 1);
    val maxsBig = tile(maxsSmall, 1, a.rows)    // maxs_big = repmat(maxs_small, [size(a, 1), 1]);
    val ret = breeze.numerics.log(sum(exp(a - maxsBig), Axis._0)) + maxsSmall

    ret.t
  }





  /**
    * function indices = argmax_over_rows(matrix)
    * @param matrix
    * @return List of indices
    */
  def argmaxOverRows(matrix:DenseMatrix[Double]):DenseVector[Int] = {
    val indices = matrix(::,*).map(dv => argmax(dv)).t // [dump, indices] = max(matrix);
    indices
  }


  /**
    * visible_state_to_hidden_probabilities(rbm_w, visible_state)
    * @param rbmWeights is a matrix of size <number of hidden units> by <number of visible units>
    * @param visibleState is a binary matrix of size <number of visible units> by <number of configurations that we're handling in parallel>.
    * @return The returned value is a matrix of size <number of hidden units> by <number of configurations that we're handling in parallel>. This takes in the (binary) states of the visible units, and returns the activation probabilities of the hidden units conditional on those states.
    */
  def visibleStateToHiddenProbabilities(rbmWeights:DenseMatrix[Double], visibleState:DenseMatrix[Double]):DenseMatrix[Double] = {
    // <solution Q3>
    logistic(rbmWeights * visibleState)
    // </solution Q3>
  }

  /**
    * visible_probability = hidden_state_to_visible_probabilities(rbm_w, hidden_state)
    * @param rbmWeights is a matrix of size <number of hidden units> by <number of visible units>
    * @param hiddenState is a binary matrix of size <number of hidden units> by <number of configurations that we're handling in parallel>.
    * @return The returned value is a matrix of size <number of visible units> by <number of configurations that we're handling in parallel>. This takes in the (binary) states of the hidden units, and returns the activation probabilities of the visible units, conditional on those states.
    */
  def hiddenStateToVisibleProbabilities(rbmWeights:DenseMatrix[Double], hiddenState:DenseMatrix[Double]):DenseMatrix[Double] = {
    // <solution Q4>
    logistic(rbmWeights.t * hiddenState)
    // </solution Q4>
  }


  /**
    * G = configuration_goodness(rbm_w, visible_state, hidden_state)
    * @param rbmWeights is a matrix of size <number of hidden units> by <number of visible units>
    * @param visibleState is a binary matrix of size <number of visible units> by <number of configurations that we're handling in parallel>.
    * @param hiddenState is a binary matrix of size <number of hidden units> by <number of configurations that we're handling in parallel>.
    * @return  the mean over cases of the goodness (negative energy) of the described configurations.
    */
  def configurationGoodnesss(rbmWeights:DenseMatrix[Double], visibleState:DenseMatrix[Double], hiddenState:DenseMatrix[Double]):Double = {
    // <solution Q5>
    sum(hiddenState * visibleState.t *:* rbmWeights) / visibleState.cols
    // </solution Q5>
  }


  /**
    * d_G_by_rbm_w = configuration_goodness_gradient(visible_state, hidden_state)
    * @param visibleState is a binary matrix of size <number of visible units> by <number of configurations that we're handling in parallel>.
    * @param hiddenState is a (possibly but not necessarily binary) matrix of size <number of hidden units> by <number of configurations that we're handling in parallel>.
    * @return  This returns the gradient of the mean configuration goodness (negative energy, as computed by function <configuration_goodness>) with respect to the model parameters. Thus, the returned value is of the same shape as the model parameters, which by the way are not provided to this function. Notice that we're talking about the mean over data cases (as opposed to the sum over data cases).
    */
  def configurationGoodnesssGradient(visibleState:DenseMatrix[Double], hiddenState:DenseMatrix[Double]):DenseMatrix[Double] = {
    // <solution Q6>
    hiddenState * visibleState.t / visibleState.cols.toDouble
    // </solution Q6>
  }


}

object RestrictedBoltzmannMachine extends LazyLogging{








}
