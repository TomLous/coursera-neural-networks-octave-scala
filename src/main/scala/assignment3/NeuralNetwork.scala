package assignment3

import breeze.linalg._
import breeze.plot._
import com.typesafe.scalalogging.LazyLogging

import scala.util.{Failure, Success, Try}

/**
  * Created by Tom Lous on 13/10/2017.
  *
  */
case class NeuralNetwork(trainingData: DataBundle, validationData: DataBundle, testData: DataBundle) extends LazyLogging{

  val numberTrainingCases: Int = trainingData.inputs.cols // n_training_cases = size(datas.training.inputs, 2);
  logger.info(s"n_training_cases $numberTrainingCases")

  /**
    *
    * function a3(wd_coefficient, n_hid, n_iters, learning_rate, momentum_multiplier, do_early_stopping, mini_batch_size)
    *
    * @param weightDecayCoefficient
    * @param numberHiddenUnits
    * @param numberIterations
    * @param learningRate
    * @param momentumMultiplier
    * @param doEarlyStopping
    * @param miniBatchSize
    * @return
    */
  def a3(id: String, weightDecayCoefficient: Double, numberHiddenUnits: Int, numberIterations: Int, learningRate: Double, momentumMultiplier: Double, doEarlyStopping: Boolean, miniBatchSize: Int): Unit = {

    val model = Model.apply(numberHiddenUnits) // model = initial_model(n_hid);

    if (numberIterations > 0) { // if n_iters ~= 0, test_gradient(model, datas.training, wd_coefficient); end
      testGradient(model, trainingData, weightDecayCoefficient)
    }

    // optimization
//    val theta:DenseVector[Double] = model.theta.thetaVector  // theta = model_to_theta(model);

    val filename = s"src/main/resources/assignment3/plots/image-loss-$id-$weightDecayCoefficient-$numberHiddenUnits-$numberIterations-$learningRate-$momentumMultiplier-$doEarlyStopping-$miniBatchSize.png"
    val name = s"$id: Loss wd: $weightDecayCoefficient #h: $numberHiddenUnits #n: $numberIterations a: $learningRate x: $momentumMultiplier s: $doEarlyStopping b: $miniBatchSize"

    val (theta ,_, trainingDataLosses, validationDataLosses, bestSoFar) =  (0 until numberIterations).foldLeft(
      (
        model.theta,
        DenseVector.zeros[Double](model.theta.thetaVector.length),
        List.empty[Double],
        List.empty[Double],
        BestSoFar(model.theta)
      )
    ){
      case ((currentTheta,currentMomentumSpeed, currentTrainingDataLosses, currentValidationDataLosses,bestSoFar), optimizationIterationI) => {
        val currentModel = currentTheta.model //model = theta_to_model(theta);

//        println(currentModel.inputToHidden)

        val trainingBatchStart = optimizationIterationI * miniBatchSize %  numberTrainingCases // training_batch_start = mod((optimization_iteration_i-1) * mini_batch_size, n_training_cases)+1;
        val trainingBatch = trainingData.batch(trainingBatchStart, miniBatchSize)

        val show = trainingData.inputs(::, 0 until miniBatchSize)
        val pos38 = trainingBatch.inputs(7, 2)
        val gradient = currentModel.dLossBydModel(trainingBatch, weightDecayCoefficient).theta.thetaVector //gradient = model_to_theta(d_loss_by_d_model(model, training_batch, wd_coefficient));

        val x = sum(gradient)

        val newMomentumSpeed = currentMomentumSpeed * momentumMultiplier - gradient // momentum_speed = momentum_speed * momentum_multiplier - gradient;

        val y = sum(newMomentumSpeed)
        val newThetaVector = currentTheta.thetaVector + currentMomentumSpeed * learningRate//theta = theta + momentum_speed * learning_rate;

        val newTheta = currentTheta.copy(thetaVector = newThetaVector) //model = theta_to_model(theta);
        val newModel = newTheta.model //model = theta_to_model(theta);

        val newTrainingDataLosses = currentTrainingDataLosses :+ newModel.loss(trainingData, weightDecayCoefficient)
        val newValidationDataLosses = currentValidationDataLosses :+ newModel.loss(validationData, weightDecayCoefficient)

        val newBestSoFar = if(doEarlyStopping && newValidationDataLosses.last < bestSoFar.validationLoss ){ //if do_early_stopping && validation_data_losses(end) < best_so_far.validation_loss,
          BestSoFar(
            newTheta, // this will be overwritten soon
            newValidationDataLosses.last, //  validation_data_losses(end);
            optimizationIterationI // optimization_iteration_i
          )
        }
        else{
          bestSoFar
        }

        val tick = Math.ceil(numberIterations / 10.0).toInt

        if(optimizationIterationI % tick == tick-1) { //if mod(optimization_iteration_i, round(n_iters/10)) == 0,
          logger.info(s"$id: After ${optimizationIterationI + 1} optimization iterations, training data loss is ${newTrainingDataLosses.last}, and validation data loss is ${newValidationDataLosses.last}")
        }



        (newTheta ,newMomentumSpeed, newTrainingDataLosses, newValidationDataLosses, newBestSoFar)

      }
    }



    // Check again, this time with more typical parameters
    if(numberIterations != 0){
      testGradient(theta.model, trainingData, weightDecayCoefficient) // if n_iters ~= 0, test_gradient(model, datas.training, wd_coefficient); end
    }

    val newTheta = if(doEarlyStopping){ //  if do_early_stopping,
      logger.info(s"$id: Early stopping: validation loss was lowest after ${bestSoFar.afterNIteratons} iterations. We chose the model that we had then.\\n'")
      bestSoFar.theta // theta = best_so_far.theta;
    }else{
      theta
    }

    // the optimization is finished. Now do some reporting.
    val newModel = newTheta.model


    if(numberIterations != 0){
      val figure = Figure(name)
      figure.clear()
      val p = figure.subplot(0)
      val x = trainingDataLosses.indices.map(_.toDouble).toArray
      p.title = "Loss"
      p.legend = true
      p += plot(x, trainingDataLosses.toArray, colorcode = "[43,146,31]", name = "training")
      p += plot(x, validationDataLosses.toArray, colorcode = "[220,0,25]", name = "validation")
      p.ylabel ="loss"
      p.xlabel = "iteration number"
      p.legend = true

      Thread.sleep(500)

      Try(figure.saveas(filename)) match {
        case Success(_) => logger.info(s"$id: Saved $filename")
        case Failure(t) => logger.error(s"$id: Couldn't save $filename: ${t.getMessage}")
      }

    }

    Map("training" -> trainingData, "validation" -> validationData, "test" -> testData).foreach{
      case (dataName, data) =>
        logger.info(s"$id: The loss on the $dataName data is "  + newModel.loss(data, weightDecayCoefficient), id)

        if(weightDecayCoefficient != 0.0){
          logger.info(s"$id: The classification loss (i.e. without weight decay) on the $dataName data is " + newModel.loss(data, 0.0))
        }

        logger.info(s"$id: The classification error rate on the $dataName data is " + newModel.classificationPerformance(data))
    }



  }


  def testGradient(model: Model, data: DataBundle, weightDecayCoefficient: Double):Unit = {
    val baseTheta:DenseVector[Double] = model.theta.thetaVector //    base_theta = model_to_theta(model);
    val h:Double = 1e-2 //    h = 1e-2;
    val correctnessThreshold: Double = 1e-5 //    correctness_threshold = 1e-5;
    val analyticGradient:DenseVector[Double] = model.dLossBydModel(data, weightDecayCoefficient).theta.thetaVector //   analytic_gradient = model_to_theta(d_loss_by_d_model(model, data, wd_coefficient));

    // Test the gradient not for every element of theta, because that's a lot of work. Test for only a few elements.

    val contributionDistances:DenseVector[Double] = new DenseVector(Array(-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0)) // contribution_distances = [-4:-1, 1:4];
    val contributionWeights:DenseVector[Double] = new DenseVector(Array(1/280.0, -4/105.0, 1/5.0, -4/5.0, 4/5.0, -1/5.0, 4/105.0, -1/280.0)) // [1/280, -4/105, 1/5, -4/5, 4/5, -1/5, 4/105, -1/280];


    (0 until 100).toStream.map(i => {
      val testIndex:Int = (i * 1299721) % baseTheta.length //test_index = mod(i * 1299721, size(base_theta,1)) + 1; % 1299721 is prime and thus ensures a somewhat random-like selection of indices
      val analyticHere:Double = analyticGradient(testIndex)
      val thetaStep = DenseVector.zeros[Double](baseTheta.length)
      thetaStep(testIndex) = h
      val temp = (0 until 8).foldLeft(0.0){
        case (tempAcc, contributionIndex) =>
          tempAcc + Theta(baseTheta + thetaStep * contributionDistances(contributionIndex)).model.loss(data, weightDecayCoefficient) * contributionWeights(contributionIndex) //temp = temp + loss(theta_to_model(base_theta + theta_step * contribution_distances(contribution_index)), data, wd_coefficient) * contribution_weights(contribution_index);
      }
      val fdHere = temp / h
      val diff = math.abs(analyticHere - fdHere)

      logger.trace(s"\ttestIndex: $testIndex, baseTheta: ${baseTheta(testIndex)}, diff: $diff, fdHere: $fdHere, analyticHere: $analyticHere")

      if(diff < correctnessThreshold || (diff / (math.abs(analyticHere) +  math.abs(fdHere))) < correctnessThreshold) {
        Left(s"Theta element #$testIndex, with value ${baseTheta(testIndex)}, has finite difference gradient $fdHere but analytic gradient $analyticHere. That looks like an error.")
      }else{
        Right(diff)
      }
    }).dropWhile(_.isRight).headOption match {
      case Some(Left(error)) => logger.error(error)
      case _ => logger.info("Gradient test passed. That means that the gradient that your code computed is within 0.001% of the gradient that the finite difference approximation computed, so the gradient calculation procedure is probably correct (not certainly, but probably).")
    }
  }

}
