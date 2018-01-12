package assignment4

import assignment3.DataBundle
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


  val trainingData = DataBundle(
    datasetMLFile.denseMatrixOption("data.training.inputs"),
    datasetMLFile.denseMatrixOption("data.training.targets")
  )

  val validationData = DataBundle(
    datasetMLFile.denseMatrixOption("data.validation.inputs"),
    datasetMLFile.denseMatrixOption("data.validation.targets")
  )

  val testData = DataBundle(
    datasetMLFile.denseMatrixOption("data.test.inputs"),
    datasetMLFile.denseMatrixOption("data.test.targets")
  )

  val randomnessSourceFileName = "/assignment4/a4_randomness_source.mat"
  val randomnessSourceFilePath = getClass.getResource(randomnessSourceFileName).toURI
  val randomnessSourceMLFile = MatLabFile(randomnessSourceFilePath)

  logger.info(s"Reading file $randomnessSourceFileName")


  val randomDataSource = RandomDataSource(randomnessSourceMLFile.denseVectorOption("randomness_source").get)


  val restrictedBoltzmannMachine = RestrictedBoltzmannMachine(trainingData, validationData, testData, randomDataSource)

  restrictedBoltzmannMachine.main("Q2.", 300, 0.0, 0.0, 0)

  restrictedBoltzmannMachine.Q3()

  restrictedBoltzmannMachine.Q4()

  restrictedBoltzmannMachine.Q5()

  restrictedBoltzmannMachine.Q6()




}
