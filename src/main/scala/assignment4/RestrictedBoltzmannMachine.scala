package assignment4

import assignment3.DataBundle
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


  def init() = {

    val test_rbm_w = randomDataSource.rand(MatrixSize(100, 256), 0.0) * 2.0 - 1.0 //    test_rbm_w = a4_rand([100, 256], 0) * 2 - 1;
    val small_test_rbm_w = randomDataSource.rand(MatrixSize(100, 256), 0.0) * 2.0 - 1.0 //small_test_rbm_w = a4_rand([10, 256], 0) * 2 - 1;

    val temp = trainingData.batch(1, 1) //    temp = extract_mini_batch(data_sets.training, 1, 1);
    val data_1_case = sampleBernoulli(temp.inputs) //    data_1_case = sample_bernoulli(temp.inputs);

    val temp2 = trainingData.batch(100, 10) //    temp = extract_mini_batch(data_sets.training, 100, 10);
    val data_10_cases = sampleBernoulli(temp.inputs) //    data_10_cases = sample_bernoulli(temp.inputs);

    val temp3 = trainingData.batch(200, 37) //   temp = extract_mini_batch(data_sets.training, 200, 37);
    val data_37_cases = sampleBernoulli(temp.inputs) //     data_37_cases = sample_bernoulli(temp.inputs);


    val test_hidden_state_1_case = sampleBernoulli(randomDataSource.rand(MatrixSize(100,1),0.0)) //    test_hidden_state_1_case = sample_bernoulli(a4_rand([100, 1], 0));
    val test_hidden_state_10_cases = sampleBernoulli(randomDataSource.rand(MatrixSize(100,10),1.0))//    test_hidden_state_10_cases = sample_bernoulli(a4_rand([100, 10], 1));
    val test_hidden_state_37_cases = sampleBernoulli(randomDataSource.rand(MatrixSize(100,37),2.0))//    test_hidden_state_37_cases = sample_bernoulli(a4_rand([100, 37], 2));
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
    val hiddenRepresentation = RestrictedBoltzmannMachine.logistic(inputToHidden * trainingData.inputs) //      hidden_representation = logistic(input_to_hid * data_sets.training.inputs);

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
        val hiddenOutput = RestrictedBoltzmannMachine.logistic(hiddenInput) //      hid_output = logistic(hid_input); %

        // size: <number of classes> by <number of data cases>
        val classificationInput = hiddenToClassification * hiddenOutput // class_input = hid_to_class * hid_output;

        // log(sum(exp of class_input)) is what we subtract to get properly normalized log class probabilities. size: <1> by <number of data cases>
        val classificationNormalizer = RestrictedBoltzmannMachine.logSumExpOverRows(classificationInput) //class_normalizer = log_sum_exp_over_rows(class_input);


        val tiledNormalizer = tile(classificationNormalizer, 1, classificationInput.rows).t

        //log of probability of each class. size: <number of classes, i.e. 10> by <number of data cases>
        val logClassificationProbability = classificationInput - tiledNormalizer // log_class_prob = class_input - repmat(class_normalizer, [size(class_input, 1), 1]);

        // Matlab has a nice method ~= to determine inequality. I just subtract the two vectors and any non-zero gets mapped to 1
        val errorRate:Double = mean((RestrictedBoltzmannMachine.argmaxOverRows(classificationInput) - RestrictedBoltzmannMachine.argmaxOverRows(data.targets)).map(x => (math.abs(x) min 1).toDouble))   // error_rate = mean(double(argmax_over_rows(class_input) ~= argmax_over_rows(data.targets))); % scalar


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

        val gradient = gradientFunction.run(currentModel, miniBatch) // gradient = gradient_function(model, mini_batch);

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
  def sampleBernoulli(probabilities:DenseMatrix[Double], reportCallsToSampleBernoulli: Boolean = false):DenseMatrix[Int] = {

    if(reportCallsToSampleBernoulli) { //if report_calls_to_sample_bernoulli,
      logger.info(s"sample_bernoulli() was called with a matrix of size ${probabilities.rows} by ${probabilities.cols}. ") //fprintf('sample_bernoulli() was called with a matrix of size %d by %d. ', size(probabilities, 1), size(probabilities, 2));
    }

    val seed:Double = sum(probabilities) //seed = sum(probabilities(:));

    val rand = randomDataSource.rand(MatrixSize(probabilities), seed)

    (probabilities >:> rand).map(b => if (b) 1 else 0) // binary = +(probabilities > a4_rand(size(probabilities), seed)); % the "+" is to avoid the "logical" data type, which just confuses things.
  }






}

object RestrictedBoltzmannMachine extends LazyLogging{


  /**
    * describe_matrix
    * @param matrix
    */
  def describeMatrix(matrix: DenseMatrix[Double]):Unit = {
    logger.info(s"Describing a matrix of size ${matrix.rows} by ${matrix.cols}. The mean of the elements is ${mean(matrix)}. The sum of the elements is ${sum(matrix)}")
    //  fprintf('Describing a matrix of size %d by %d. The mean of the elements is %f. The sum of the elements is %f\n', size(matrix, 1), size(matrix, 2), mean(matrix(:)), sum(matrix(:)))
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


}
