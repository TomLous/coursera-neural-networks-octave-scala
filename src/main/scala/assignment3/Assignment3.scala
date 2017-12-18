package assignment3

import java.util.Calendar

import com.typesafe.scalalogging.LazyLogging
import io.MatLabFile


object Assignment3 extends App with LazyLogging {

  val inputFileName = "/assignment3/data.mat"
  val inputFilePath = getClass.getResource(inputFileName).toURI
  val mlFile = MatLabFile(inputFilePath)

  logger.info(s"Reading file $inputFileName")


  val trainingData = DataBundle(
    mlFile.denseMatrixOption("data.training.inputs"),
    mlFile.denseMatrixOption("data.training.targets")
  )

  val validationData = DataBundle(
    mlFile.denseMatrixOption("data.validation.inputs"),
    mlFile.denseMatrixOption("data.validation.targets")
  )

  val testData = DataBundle(
    mlFile.denseMatrixOption("data.test.inputs"),
    mlFile.denseMatrixOption("data.test.targets")
  )


  val nn = NeuralNetwork(trainingData, validationData, testData)



  // Q2
  exercise("Q2. What is the loss on the training data for that test run? Write your answer with at least 5 digits after the decimal point.")(
    () => nn.a3(0, 0, 0, 0, 0, false, 0)
  )


  def exercise(info: String)(f:() => Unit) = {
    logger.info("_" * 80)
    logger.info(info)
    val startTime:Long = Calendar.getInstance().getTimeInMillis
    f()
    val elapsed:Long = Calendar.getInstance().getTimeInMillis - startTime
    logger.info(s"$elapsed ms elapsed")
    logger.info("_" * 80)

  }

}
