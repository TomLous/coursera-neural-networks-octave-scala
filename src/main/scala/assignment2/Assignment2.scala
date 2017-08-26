package assignment2


import com.typesafe.scalalogging.LazyLogging
import io.MatLabFile

/**
  * Created by Tom Lous on 26/08/2017.
  */
object Assignment2  extends App with LazyLogging{

  val inputFileName = "/assignment2/data.mat"
  val inputFilePath = getClass.getResource(inputFileName).toURI
  val mlFile = MatLabFile(inputFilePath)

  logger.info(s"Reading file $inputFileName")

  val neuralNetwork:Either[String, NeuralNetwork]  = (
    mlFile.denseMatrixOption("data.trainData"),
    mlFile.denseMatrixOption("data.validData"),
    mlFile.denseMatrixOption("data.testData"),
    mlFile.stringListOption("data.vocab")) match {
    case (Some(trainData), Some(validData), Some(testData), Some(vocab)) =>
      Right(NeuralNetwork("Model1", trainData,validData,testData,vocab))
    case _ =>
      Left("Not all samples found")
  }

  val model = neuralNetwork.map(_.train(epochs = 1))




//  NeuralNetwork(trainingData)


//  println(trainingData)
//  println(validData)
//  println(testData)
//  println(vocab)


  /**
    * numdims = size(data.trainData, 1);
D = numdims - 1;
M = floor(size(data.trainData, 2) / N);
train_input = reshape(data.trainData(1:D, 1:N * M), D, N, M);
train_target = reshape(data.trainData(D + 1, 1:N * M), 1, N, M);
valid_input = data.validData(1:D, :);
valid_target = data.validData(D + 1, :);
test_input = data.testData(1:D, :);
test_target = data.testData(D + 1, :);
vocab = data.vocab;
     */
//  val mlFile.

}
