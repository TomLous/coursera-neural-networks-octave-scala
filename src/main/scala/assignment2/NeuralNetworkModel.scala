package assignment2

import breeze.linalg.DenseMatrix

/**
  * Created by Tom Lous on 26/08/2017.
  */
case class NeuralNetworkModel(
                               word_embedding_weights: DenseMatrix[Double],
                               embed_to_hid_weights: DenseMatrix[Double],
                               hid_to_output_weights: DenseMatrix[Double],
                               hid_bias: DenseMatrix[Double],
                               output_bias: DenseMatrix[Double],
                               vocab: List[String]
                             ) {



}
