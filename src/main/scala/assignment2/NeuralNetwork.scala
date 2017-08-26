package assignment2

import breeze.linalg._
import breeze.numerics._
import com.typesafe.scalalogging.LazyLogging
import util.DenseMatrixUtils._



/**
  * Created by Tom Lous on 26/08/2017.
  * Copyright © 2017 Datlinq B.V..
  */
case class NeuralNetwork(
                          name: String,
                          trainingData: DenseMatrix[Double],
                          validationData: DenseMatrix[Double],
                          testData: DenseMatrix[Double],
                          vocabulary: List[String],
                          batchSize: Int = 100
                        ) extends LazyLogging {


  // LOAD DATA
  private lazy val (train_input, train_target) = restructureInputMatrix(trainingData)
  private lazy val (valid_input, valid_target) = restructureInputMatrix(validationData)
  private lazy val (test_input, test_target) = restructureInputMatrix(testData)

  private lazy val numwords = train_input.head.rows
  private lazy val numbatches = train_input.size
  private lazy val vocab_size = vocabulary.size


  /**
    * This function trains a neural network language model.
    *
    * @param epochs        Number of epochs to run.
    * @param learning_rate Learning rate; default = 0.1.
    * @param momentum      Momentum; default = 0.9.
    * @param numhid1       Dimensionality of embedding space; default = 50.
    * @param numhid2       Number of units in hidden layer; default = 200.
    * @param init_wt       Standard deviation of the normal distribution which is sampled to get the initial weights; default = 0.01
    * @return NeuralNetwork
    */
  def train(epochs: Int,
            learning_rate: Double = 0.1,
            momentum: Double = 0.9,
            numhid1: Int = 50,
            numhid2: Int = 200,
            init_wt: Double = 0.01): NeuralNetworkModel = {

    val start_time = System.currentTimeMillis()
    val show_training_CE_after = 100
    val show_validation_CE_after = 1000

    // INITIALIZE WEIGHTS AND BIASES.
    val word_embedding_weights: DenseMatrix[Double] = init_wt * DenseMatrix.rand[Double](vocab_size, numhid1)
    val embed_to_hid_weights: DenseMatrix[Double] = init_wt * DenseMatrix.rand[Double](numwords * numhid1, numhid2)
    val hid_to_output_weights: DenseMatrix[Double] = init_wt * DenseMatrix.rand[Double](numhid2, vocab_size)
    val hid_bias: DenseMatrix[Double] = DenseMatrix.zeros[Double](numhid2, 1)
    val output_bias: DenseMatrix[Double] = DenseMatrix.zeros[Double](vocab_size, 1)


    val word_embedding_weights_delta = DenseMatrix.zeros[Double](vocab_size, numhid1)
    val word_embedding_weights_gradient = DenseMatrix.zeros[Double](vocab_size, numhid1)
    val embed_to_hid_weights_delta = DenseMatrix.zeros[Double](numwords * numhid1, numhid2)
    val hid_to_output_weights_delta = DenseMatrix.zeros[Double](numhid2, vocab_size)
    val hid_bias_delta = DenseMatrix.zeros[Double](numhid2, 1)
    val output_bias_delta = DenseMatrix.zeros[Double](vocab_size, 1)
    val expansion_matrix = DenseMatrix.eye[Double](vocab_size)
    val count = 0
    val tiny = Math.exp(-30)


    // TRAIN.
    (1 to epochs).map(epoch => {
      logger.info(s"Epoch $epoch")
      val this_chunk_CE = 0
      val trainset_CE = 0

      // LOOP OVER MINI-BATCHES.
      (0 until numbatches).map(m => {
        val input_batch = train_input(m)
        val target_batch = train_target(m)

        // FORWARD PROPAGATE.
        // Compute the state of each layer in the network given the input batch
        // and all weights and biases
        val (embedding_layer_state, hidden_layer_state, output_layer_state) =
        fprop(input_batch, word_embedding_weights, embed_to_hid_weights,
          hid_to_output_weights, hid_bias, output_bias)
        4
      })


    })


    NeuralNetworkModel(1, .1, .2, .3, .4, vocabulary)
  }


  /**
    * This method forward propagates through a neural network.
    *
    * @param input_batch            The input data as a matrix of size numwords X batchsize where,
    *                               numwords is the number of words, batchsize is the number of data points.
    *                               So, if input_batch(i, j) = k then the ith word in data point j is word
    *                               index k of the vocabulary.
    * @param word_embedding_weights Word embedding as a matrix of size
    *                               vocab_size X numhid1, where vocab_size is the size of the vocabulary
    *                               numhid1 is the dimensionality of the embedding space.
    * @param embed_to_hid_weights   Weights between the word embedding layer and hidden
    *                               layer as a matrix of soze numhid1*numwords X numhid2, numhid2 is the
    *                               number of hidden units.
    * @param hid_to_output_weights  Weights between the hidden layer and output softmax
    *                               unit as a matrix of size numhid2 X vocab_size
    * @param hid_bias               Bias of the hidden layer as a matrix of size numhid2 X 1.
    * @param output_bias            Bias of the output layer as a matrix of size vocab_size X 1.
    * @return (
    *         embedding_layer_state: State of units in the embedding layer as a matrix of
    *         size numhid1*numwords X batchsize
    *         ,
    *         hidden_layer_state: State of units in the hidden layer as a matrix of size
    *         numhid2 X batchsize
    *         ,
    *         output_layer_state: State of units in the output layer as a matrix of size
    *         vocab_size X batchsize
    */
  private def fprop(input_batch: DenseMatrix[Double],
                    word_embedding_weights: DenseMatrix[Double],
                    embed_to_hid_weights: DenseMatrix[Double],
                    hid_to_output_weights: DenseMatrix[Double],
                    hid_bias: DenseMatrix[Double],
                    output_bias: DenseMatrix[Double]): (DenseMatrix[Double], DenseMatrix[Double], DenseMatrix[Double]) = {
    val numhid1 = word_embedding_weights.cols
    val numhid2 = embed_to_hid_weights.cols


    // COMPUTE STATE OF WORD EMBEDDING LAYER.
    //  Look up the inputs word indices in the word_embedding_weights matrix.
    val embedding_layer_state = word_embedding_weights(
      input_batch
        .reshape(1)
        .toArray
        .map(_.toInt - 1) // indices based on matlab => -1
        .toIndexedSeq
      , ::)
      .toDenseMatrix
      .reshape(numhid1 * numwords)

    // COMPUTE STATE OF HIDDEN LAYER.
    //  Compute inputs to hidden units.
    val inputs_to_hidden_units = (embed_to_hid_weights.t * embedding_layer_state) + tile(hid_bias, 1, batchSize)


    // Apply logistic activation function.
    //  % FILL IN CODE. Replace the line below by one of the options.
    // Options:
    // (a) hidden_layer_state = 1 ./ (1 + exp(inputs_to_hidden_units));
    // (b) hidden_layer_state = 1 ./ (1 - exp(-inputs_to_hidden_units));
    // => (c) hidden_layer_state = 1 ./ (1 + exp(-inputs_to_hidden_units));
    // (d) hidden_layer_state = -1 ./ (1 + exp(-inputs_to_hidden_units));
//    val hidden_layer_state = DenseMatrix.zeros[Double](numhid2, batchSize)
    val hidden_layer_state = 1.0 ./ (exp(-inputs_to_hidden_units) + 1.0)


    // COMPUTE STATE OF OUTPUT LAYER.
    //  Compute inputs to softmax.
    //  % FILL IN CODE. Replace the line below by one of the options.
    // Options:
    // => (a) inputs_to_softmax = hid_to_output_weights' * hidden_layer_state +  repmat(output_bias, 1, batchsize);
    // (b) inputs_to_softmax = hid_to_output_weights' * hidden_layer_state +  repmat(output_bias, batchsize, 1);
    // (c) inputs_to_softmax = hidden_layer_state * hid_to_output_weights' +  repmat(output_bias, 1, batchsize);
    // (d) inputs_to_softmax = hid_to_output_weights * hidden_layer_state +  repmat(output_bias, batchsize, 1);
//    val inputs_to_softmax = DenseMatrix.zeros[Double](vocab_size, batchSize)
    val inputs_to_softmax = (hid_to_output_weights.t * hidden_layer_state) + tile(output_bias, 1, batchSize)


    // Subtract maximum.
    // Remember that adding or subtracting the same constant from each input to a
    // softmax unit does not affect the outputs. Here we are subtracting maximum to
    // make all inputs <= 0. This prevents overflows when computing their
    // exponents.
    val inputs_to_softmax_norm = inputs_to_softmax - tile(max(inputs_to_softmax, Axis._0).t.toDenseMatrix, vocab_size, 1)

    // Compute exp.
    val output_layer_state = exp(inputs_to_softmax_norm)

    // Normalize to get probability distribution.
    val output_layer_state_norm = output_layer_state /:/ tile(sum(output_layer_state, Axis._0).t.toDenseMatrix, vocab_size, 1)


    assertMatrixDimensions("embedding_layer_state", embedding_layer_state, numhid1*numwords, batchSize)
    assertMatrixDimensions("hidden_layer_state", hidden_layer_state, numhid2, batchSize)
    assertMatrixDimensions("output_layer_state_norm", output_layer_state_norm,vocab_size, batchSize)

    (embedding_layer_state, hidden_layer_state, output_layer_state_norm)
  }


  def assertMatrixDimensions(name: String, matrix: DenseMatrix[Double], expectedRows:Int, expectedCols:Int):Unit = {
    val rows = matrix.rows
    val cols = matrix.cols

    assert(rows == expectedRows && cols == expectedCols, s"Unexpected dimensions for $name [$rows x $cols] = expected [$expectedRows x $expectedCols]")

  }

  /**
    * Splits D * X matrix into array of X/N batches (matrices) of D-1 * N size & 1 * N size. Using last row as target (y)
    *
    * @param matrix DenseMatrix[Double]
    * @return Tuple of input & target lists of matrices
    */
  private def restructureInputMatrix(matrix: DenseMatrix[Double]): (List[DenseMatrix[Double]], List[DenseMatrix[Double]]) = {
    val D: Int = matrix.rows - 1
    val N: Int = batchSize
    val M: Int = Math.floor(matrix.cols / N.toDouble).toInt

    (
      hsplit(matrix(0 until D, 0 until (N * M)), M).toList,
      hsplit(matrix(D to D, 0 until (N * M)), M).toList
    )
  }


}
