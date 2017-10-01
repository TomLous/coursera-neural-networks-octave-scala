package assignment2


import com.typesafe.scalalogging.LazyLogging
import io.MatLabFile
import util.DynamicFileLogging

/**
  * Created by Tom Lous on 26/08/2017.
  * @todo can't get the CE errors to emulate octave's CE errors. Unfinished
  * @todo split logging somehow
  */
object Assignment2 extends App with LazyLogging {

//  val logPath = getClass.getResource("/assignment2/logs/").toString
  val logPath = s"src/main/resources/assignment2/logs/"

  val inputFileName = "/assignment2/data.mat"
  val inputFilePath = getClass.getResource(inputFileName).toURI
  val mlFile = MatLabFile(inputFilePath)

  logger.info(s"Reading file $inputFileName")

  val neuralNetwork: Either[String, NeuralNetwork] = (
    mlFile.denseMatrixOption("data.trainData"),
    mlFile.denseMatrixOption("data.validData"),
    mlFile.denseMatrixOption("data.testData"),
    mlFile.stringListOption("data.vocab")) match {
    case (Some(trainData), Some(validData), Some(testData), Some(vocab)) =>
      Right(NeuralNetwork("Model1", trainData, validData, testData, vocab))
    case _ =>
      Left("Not all samples found")
  }

  //  val modelQ2 = neuralNetwork.map(_.train(epochs = 10))



  val experiments = List(
    TrainingCase("Question 2", epochs = 10),
    TrainingCase("Question 3", epochs = 10, learning_rate = 100.0),
    TrainingCase("Question 5 A", epochs = 1, learning_rate = 0.001),
    TrainingCase("Question 5 B", epochs = 1),
    TrainingCase("Question 5 C", epochs = 1, learning_rate = 10.0),
    TrainingCase("Question 6 A", epochs = 10, learning_rate = 0.001),
    TrainingCase("Question 6 B",epochs = 10),
    TrainingCase("Question 6 C", epochs = 10, learning_rate = 10.0),
    TrainingCase("Question 7 A", epochs = 10, numhid1 = 5, numhid2 = 100),
    TrainingCase("Question 7 B", epochs = 10, numhid2 = 10),
    TrainingCase("Question 7 C", epochs = 10),
    TrainingCase("Question 7 D", epochs = 10, numhid1 = 100, numhid2 = 5),
    TrainingCase("Question 9 A", epochs = 5, momentum = 0.0),
    TrainingCase("Question 9 B", epochs = 5, momentum = 0.5),
    TrainingCase("Question 9 C", epochs = 5)
  )


  val models = neuralNetwork.map(nn => {
    experiments.par.map(tc => {
      val logger = DynamicFileLogging(logPath + tc.fileName).logger
      val nnModel = nn.train(tc)(logger)
      nnModel
    })
  })


}
