package assignment2

import breeze.linalg.{DenseMatrix, hsplit}
import com.typesafe.scalalogging.LazyLogging

/**
  * Created by Tom Lous on 26/08/2017.
  * Copyright © 2017 Datlinq B.V..
  */
case class NeuralNetwork(
                       name: String,
                       trainingData: DenseMatrix[Double],
                       validationData: DenseMatrix[Double],
                       testData: DenseMatrix[Double],
                       vocabulary: List[String],
                       batchSize: Int = 100
                     ) extends LazyLogging {

//  val D:Int = trainingData.rows - 1
//  val N:Int = batchSize
//  val M:Int = Math.floor(trainingData.cols / N.toDouble).toInt

  private val (train_input, train_target) = restructureInputMatrix(trainingData)
  private val (valid_input, valid_target) = restructureInputMatrix(validationData)
  private val (test_input,  test_target) = restructureInputMatrix(testData)


  /**
    * This function trains a neural network language model.
    * @param epochs Number of epochs to run.
    * @param learning_rate Learning rate; default = 0.1.
    * @param momentum Momentum; default = 0.9.
    * @param numhid1 Dimensionality of embedding space; default = 50.
    * @param numhid2 Number of units in hidden layer; default = 200.
    * @param init_wt Standard deviation of the normal distribution which is sampled to get the initial weights; default = 0.01
    * @return NeuralNetwork
    */
  def train(epochs:Int,
            learning_rate: Double=0.1,
            momentum: Double=0.9,
            numhid1: Int=50,
            numhid2: Int=200,
            init_wt: Double=0.01):NeuralNetworkModel = {

    val start_time = System.currentTimeMillis()


    NeuralNetworkModel(1,.1,.2,.3,.4,vocabulary)
  }

  /**
    * Splits D * X matrix into array of X/N batches (matrices) of D-1 * N size & 1 * N size. Using last row as target (y)
    * @param matrix DenseMatrix[Double]
    * @return Tuple of input & target lists of matrices
    */
  private def restructureInputMatrix(matrix: DenseMatrix[Double]):(List[DenseMatrix[Double]],List[DenseMatrix[Double]]) = {
    val D:Int = matrix.rows - 1
    val N:Int = batchSize
    val M:Int = Math.floor(matrix.cols / N.toDouble).toInt

    (
      hsplit(matrix(0 until D, 0 until (N*M)), M).toList,
      hsplit(matrix(D to D, 0 until (N*M)), M).toList
    )
  }


}
