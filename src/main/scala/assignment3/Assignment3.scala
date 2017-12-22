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
/*
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
*/
  exercise("Q3c.  best see the effect of the optimization")(
    () => nn.a3("Q3c", 0, 10, 1, 0.005, 0, false, 4)
  )
  /*
    exercise("Q4a.  Let's try a bigger learning rate: LR=0.5, and still no momentum.")(
      () => nn.a3("Q4a", 0, 10, 70, 0.5, 0, false, 4)
    )

    exercise("Q4b.  0.002, 0.01, 0.05, 0.2, 1.0, 5.0, and 20.0")(
      () => {
        nn.a3("Q4b-1", 0, 10, 70, 0.002, 0, false, 4)
        nn.a3("Q4b-2", 0, 10, 70, 0.01, 0, false, 4)
        nn.a3("Q4b-3", 0, 10, 70, 0.05, 0, false, 4)
        nn.a3("Q4b-4", 0, 10, 70, 0.2, 0, false, 4)
        nn.a3("Q4b-5", 0, 10, 70, 1.0, 0, false, 4)
        nn.a3("Q4b-6", 0, 10, 70, 5.0, 0, false, 4)
        nn.a3("Q4b-7", 0, 10, 70, 20.0, 0, false, 4)
        nn.a3("Q4b-8", 0, 10, 70, 0.002, 0.9, false, 4)
        nn.a3("Q4b-9", 0, 10, 70, 0.01, 0.9, false, 4)
        nn.a3("Q4b-10", 0, 10, 70, 0.05, 0.9, false, 4)
        nn.a3("Q4b-11", 0, 10, 70, 0.2, 0.9, false, 4)
        nn.a3("Q4b-12", 0, 10, 70, 1.0, 0.9, false, 4)
        nn.a3("Q4b-13", 0, 10, 70, 5.0, 0.9, false, 4)
        nn.a3("Q4b-14", 0, 10, 70, 20.0, 0.9, false, 4)
      }
    )


    exercise("Q6.  Generalization")(
      () => nn.a3("Q6", 0, 200, 1000, 0.35, 0.9, false, 100)
    )


    exercise("Q7.  Early Stopping")(
      () => nn.a3("Q7", 0, 200, 1000, 0.35, 0.9, true, 100)
    )

    exercise("Q8.  Weight Decay")(
      () => {
        nn.a3("Q8-1", 5, 200, 1000, 0.35, 0.9, false, 100)
        nn.a3("Q8-2", 0.01, 200, 1000, 0.35, 0.9, false, 100)
        nn.a3("Q8-3", 1, 200, 1000, 0.35, 0.9, false, 100)
        nn.a3("Q8-4", 0, 200, 1000, 0.35, 0.9, false, 100)
        nn.a3("Q8-5", 0.001, 200, 1000, 0.35, 0.9, false, 100)
        nn.a3("Q8-6", 0.0001, 200, 1000, 0.35, 0.9, false, 100)
      }
    )

    exercise("Q9. Reducing the number of hidden units.")(
      () => {
        nn.a3("Q9-1", 0, 30, 1000, 0.35, 0.9, false, 100)
        nn.a3("Q9-2", 0, 130, 1000, 0.35, 0.9, false, 100)
        nn.a3("Q9-3", 0, 170, 1000, 0.35, 0.9, false, 100)
        nn.a3("Q9-4", 0, 100, 1000, 0.35, 0.9, false, 100)
        nn.a3("Q9-5", 0, 10, 1000, 0.35, 0.9, false, 100)
      }
    )


    exercise("Q10. Reducing the number of hidden units. With Early stopping")(
      () => {
        nn.a3("Q10-1", 0, 18, 1000, 0.35, 0.9, true, 100)
        nn.a3("Q10-2", 0, 236, 1000, 0.35, 0.9, true, 100)
        nn.a3("Q10-3", 0, 83, 1000, 0.35, 0.9, true, 100)
        nn.a3("Q10-4", 0, 37, 1000, 0.35, 0.9, true, 100)
        nn.a3("Q10-5", 0, 113, 1000, 0.35, 0.9, true, 100)
      }
    )
   */






  def exercise(info: String)(f:() => Unit):Unit = {
    logger.info(info)
    val startTime:Long = Calendar.getInstance().getTimeInMillis
    f()
    val elapsed:Long = Calendar.getInstance().getTimeInMillis - startTime
    logger.info(s"$elapsed ms elapsed")
    logger.info("_" * 80)
  }

  System.exit(0)
}
