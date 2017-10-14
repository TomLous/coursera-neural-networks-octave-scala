package assignment3

/**
  * Created by Tom Lous on 13/10/2017.
  *
  */
case class NeuralNetwork(trainingData: DataBundle, validationData: DataBundle, testData: DataBundle) {

  val numberTrainingCases: Int = trainingData.inputs.cols

  /**
    *
    * function a3(wd_coefficient, n_hid, n_iters, learning_rate, momentum_multiplier, do_early_stopping, mini_batch_size)
    *
    * @param wdCoefficient
    * @param numberHiddenUnits
    * @param numberIterations
    * @param learningRate
    * @param momentumMultiplier
    * @param doEarlyStopping
    * @param iniBatchSize
    * @return
    */
  def a3(wdCoefficient: Double, numberHiddenUnits: Int, numberIterations: Int, learningRate: Double, momentumMultiplier: Double, doEarlyStopping: Boolean, iniBatchSize: Int): Unit = {




    val model = Model.apply(numberHiddenUnits)

    if (numberIterations > 0) {

    }


  }


  def testGradient(model: Model, data: DataBundle, wdCoefficient: Double) = {

  }

}
