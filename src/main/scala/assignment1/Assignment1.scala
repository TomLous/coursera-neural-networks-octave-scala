package assignment1

import com.typesafe.scalalogging.LazyLogging
import io.MatLabFile

/**
  * Created by Tom Lous on 14/08/2017.
  *
  */
object Assignment1 extends App with LazyLogging{

  System.setProperty("log.assignment", "assignment1")

  val learningRate = 1.0

  // define matlab files for training Perceptrons
  val resources = List(
    "/assignment1/dataset1.mat",
    "/assignment1/dataset2.mat",
    "/assignment1/dataset3.mat",
    "/assignment1/dataset4.mat"
  )

  resources.map(inputFileName => {

    logger.info(s"Reading file $inputFileName")

    val inputFilePath = getClass.getResource(inputFileName).toURI
    val mlFile = MatLabFile(inputFilePath)

    // get results
    val results = ( // load certain matrices from matlab file
      mlFile.denseMatrixOption("neg_examples_nobias"),
      mlFile.denseMatrixOption("pos_examples_nobias"),
      mlFile.denseVectorOption("w_init"),
      mlFile.denseVectorOption("w_gen_feas")
    ) match {
      case (Some(neg_examples_nobias), Some(pos_examples_nobias), w_init_opt, w_gen_feas_opt) => // if found => train Perceptron
        Perceptron(inputFileName.substring(inputFileName.lastIndexOf("/")+1,inputFileName.lastIndexOf(".")), neg_examples_nobias, pos_examples_nobias,w_init_opt, w_gen_feas_opt, learningRate).learn()
      case _ =>
        Left("Not all samples found")
    }


    println(results)
  })
}
