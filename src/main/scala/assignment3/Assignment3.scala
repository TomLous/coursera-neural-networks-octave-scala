package assignment3

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

  nn.a3(0, 0, 0, 0, 0, false, 0)

//  nn.a3()



//  println(x.map(_.cols))
}
