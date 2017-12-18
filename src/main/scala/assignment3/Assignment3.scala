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

  logger.info("_" * 80)

  // Q2
  exercise("Q2. What is the loss on the training data for that test run? Write your answer with at least 5 digits after the decimal point.")(
    () => nn.a3("Q2", 0, 0, 0, 0, 0, false, 0)
  )

 // Q3
  exercise("Q3a.  run with huge weight decay, so that the weight decay loss overshadows the classification loss. ")(
    () => nn.a3("Q3a", 1e7, 7, 10, 0, 0, false, 4)
  )


  exercise("Q3b.  turn off weight decay, and you'll see the gradient error message coming back ")(
    () => nn.a3("Q3b", 0, 7, 10, 0, 0, false, 4)
  )




  def exercise(info: String)(f:() => Unit):Unit = {
    logger.info(info)
    val startTime:Long = Calendar.getInstance().getTimeInMillis
    f()
    val elapsed:Long = Calendar.getInstance().getTimeInMillis - startTime
    logger.info(s"$elapsed ms elapsed")
    logger.info("_" * 80)
  }
}
