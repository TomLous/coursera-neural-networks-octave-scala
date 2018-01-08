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

  val dataset = datasetMLFile.content("data")

  val randomnessSourceFileName = "/assignment4/a4_randomness_source.mat"
  val randomnessSourceFilePath = getClass.getResource(randomnessSourceFileName).toURI
  val randomnessSourceMLFile = MatLabFile(randomnessSourceFilePath)

  logger.info(s"Reading file $randomnessSourceFileName")


  val randomDataSource = RandomDataSource(randomnessSourceMLFile.denseVectorOption("randomness_source").get)

  println(dataset)
  println(randomDataSource.randomSource(0 to 20))


}
