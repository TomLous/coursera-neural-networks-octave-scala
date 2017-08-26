package assignment2

/**
  * Created by Tom Lous on 26/08/2017.
  */
case class NeuralNetworkModel(
                               word_embedding_weights: Int,
                               embed_to_hid_weights: Double,
                               hid_to_output_weights: Double,
                               hid_bias: Double,
                               output_bias: Double,
                               vocab: List[String]
                             ) {

}
