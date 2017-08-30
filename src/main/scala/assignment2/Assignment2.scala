package assignment2


import com.typesafe.scalalogging.LazyLogging
import io.MatLabFile
import util.DynamicFileLogging

/**
  * Created by Tom Lous on 26/08/2017.
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
    TrainingCase("Question 5 B", epochs = 1, learning_rate = 0.1)
//    TrainingCase(epochs = 1, learning_rate = 10.0), // Question 5 C
//    TrainingCase(epochs = 10, learning_rate = 0.001), // Question 6 A
//    TrainingCase(epochs = 10, learning_rate = 0.1), // Question 6 B
//    TrainingCase(epochs = 10, learning_rate = 10.0), // Question 6 C
//    TrainingCase(epochs = 10, numhid1 = 5, numhid2 = 100), // Question 7 A
//    TrainingCase(epochs = 10, numhid1 = 50, numhid2 = 10), // Question 7 B
//    TrainingCase(epochs = 10, numhid1 = 50, numhid2 = 200), // Question 7 C
//    TrainingCase(epochs = 10, numhid1 = 100, numhid2 = 5), // Question 7 D,
//    TrainingCase(epochs = 5, momentum = 0.0), // Question 9 A,
//    TrainingCase(epochs = 5, momentum = 0.5), // Question 9 B,
//    TrainingCase(epochs = 5, momentum = 0.9), // Question 9 C,
  )


  val models = neuralNetwork.map(nn => {
    experiments.par.map(tc => {
      val logger = DynamicFileLogging(logPath + tc.fileName).logger
      val nnModel = nn.train(tc)(logger)
      nnModel
    })
  })


  //  NeuralNetwork(trainingData)


  //  println(trainingData)
  //  println(validData)
  //  println(testData)
  //  println(vocab)


  /**
    * numdims = size(data.trainData, 1);
    * D = numdims - 1;
    * M = floor(size(data.trainData, 2) / N);
    * train_input = reshape(data.trainData(1:D, 1:N * M), D, N, M);
    * train_target = reshape(data.trainData(D + 1, 1:N * M), 1, N, M);
    * valid_input = data.validData(1:D, :);
    * valid_target = data.validData(D + 1, :);
    * test_input = data.testData(1:D, :);
    * test_target = data.testData(D + 1, :);
    * vocab = data.vocab;
    */
  //  val mlFile.

}
