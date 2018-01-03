package assignment4

import com.typesafe.scalalogging.LazyLogging
import io.MatLabFile

/**
  * Created by Tom Lous on 03/01/2018.
  */
object Assignment4 extends App with LazyLogging {

  System.setProperty("log.assignment", "assignment4")

  val datasetFileName = "/assignment4/data_set.mat"
  val datasetFilePath = getClass.getResource(datasetFileName).toURI
  val datasetMLFile = MatLabFile(datasetFilePath)

  logger.info(s"Reading file $datasetFileName")

  val randomSourceFileName = "/assignment4/a4_randomness_source.mat"
  val randomSourceFilePath = getClass.getResource(randomSourceFileName).toURI
  val randomSourceMLFile = MatLabFile(randomSourceFilePath)

  logger.info(s"Reading file $randomSourceFileName")

}
