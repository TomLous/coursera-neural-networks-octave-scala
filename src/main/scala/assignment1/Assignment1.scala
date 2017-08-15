package assignment1

import com.typesafe.scalalogging.LazyLogging
import io.MatLabFile

/**
  * Created by Tom Lous on 14/08/2017.
  * Copyright © 2017 Datlinq B.V..
  */
object Assignment1 extends App with LazyLogging{

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

    val perceptron = (
      mlFile.denseMatrixOption("neg_examples_nobias"),
      mlFile.denseMatrixOption("pos_examples_nobias"),
      mlFile.denseVectorOption("w_init"),
      mlFile.denseVectorOption("w_gen_feas")
    ) match {
      case (Some(neg_examples_nobias), Some(pos_examples_nobias), w_init_opt, w_gen_feas_opt) =>
        Perceptron(neg_examples_nobias, pos_examples_nobias,w_init_opt, w_gen_feas_opt).learn()
      case _ =>
        Left("Not all samples found")
    }


    println(perceptron)
  })
}
