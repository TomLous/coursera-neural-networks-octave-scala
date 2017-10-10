package assignment3

import com.typesafe.scalalogging.LazyLogging
import io.MatLabFile


object Assignment3 extends App with LazyLogging {

  val inputFileName = "/assignment3/data.mat"
  val inputFilePath = getClass.getResource(inputFileName).toURI
  val mlFile = MatLabFile(inputFilePath)

  logger.info(s"Reading file $inputFileName")

  println(mlFile.listNames)
  val x = mlFile.denseMatrixOption("data.training.inputs")
  val y = mlFile.denseMatrixOption("data.training")
  val z = mlFile.denseMatrixOption("data.validation")
  val w = mlFile.denseMatrixOption("data.test")

  println(x)
}
