package assignment1

import breeze.linalg.{DenseMatrix, DenseVector}

/**
  * Created by Tom Lous on 13/08/2017.
  */
case class LearnPerceptron(
                            neg_examples_nobias: DenseMatrix[Double],
                            pos_examples_nobias: DenseMatrix[Double],
                            w_init: Option[DenseVector[Double]],
                            w_gen_feas: Option[DenseVector[Double]]
                          ) {





}
