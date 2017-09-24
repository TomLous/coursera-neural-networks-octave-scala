package assignment2

import breeze.linalg._
import breeze.numerics._
import com.typesafe.scalalogging.Logger
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
                          batchsize: Int = 100
                        ) {


  // LOAD DATA
  private lazy val (train_input, train_target) = restructureInputMatrixBatch(trainingData)
  private lazy val (valid_input, valid_target) = restructureInputMatrix(validationData)
  private lazy val (test_input, test_target) = restructureInputMatrix(testData)

  private lazy val numwords = train_input.head.rows
  private lazy val numbatches = train_input.size
  private lazy val vocab_size = vocabulary.size


  /**
    * This function trains a neural network language model.
    *
    * @return NeuralNetwork
    */
  def train(trainingCase: TrainingCase)(logger: Logger): NeuralNetworkModel = {
    val epochs = trainingCase.epochs
    val learning_rate = trainingCase.learning_rate
    val momentum = trainingCase.momentum
    val numhid1 = trainingCase.numhid1
    val numhid2 = trainingCase.numhid2
    val init_wt = trainingCase.init_wt

    val start_time = System.currentTimeMillis()
    val show_training_CE_after = 100
    val show_validation_CE_after = 1000

    // INITIALIZE WEIGHTS AND BIASES.
    var word_embedding_weights: DenseMatrix[Double] = init_wt * DenseMatrix.rand[Double](vocab_size, numhid1)
    var embed_to_hid_weights: DenseMatrix[Double] = init_wt * DenseMatrix.rand[Double](numwords * numhid1, numhid2)
    var hid_to_output_weights: DenseMatrix[Double] = init_wt * DenseMatrix.rand[Double](numhid2, vocab_size)
    var hid_bias: DenseMatrix[Double] = DenseMatrix.zeros[Double](numhid2, 1)
    var output_bias: DenseMatrix[Double] = DenseMatrix.zeros[Double](vocab_size, 1)


    var word_embedding_weights_delta = DenseMatrix.zeros[Double](vocab_size, numhid1)
    val word_embedding_weights_gradient_init = DenseMatrix.zeros[Double](vocab_size, numhid1)
    var embed_to_hid_weights_delta = DenseMatrix.zeros[Double](numwords * numhid1, numhid2)
    var hid_to_output_weights_delta = DenseMatrix.zeros[Double](numhid2, vocab_size)
    var hid_bias_delta = DenseMatrix.zeros[Double](numhid2, 1)
    var output_bias_delta = DenseMatrix.zeros[Double](vocab_size, 1)
    val expansion_matrix = DenseMatrix.eye[Double](vocab_size)
    val tiny = Math.exp(-30)

    var count = 0
    var trainset_CE = 0.0


    def check(title: String, input: DenseMatrix[Double], target: DenseMatrix[Double]):Unit = {
      logger.info(s"$trainingCase: Running $title ...")
      val (_, _, output_layer_state_valid) =
        fprop(input, word_embedding_weights, embed_to_hid_weights, hid_to_output_weights, hid_bias, output_bias)

      val expanded_valid_target = expansion_matrix(::, target.toIndexedSequence())

      val CE = -sum(expanded_valid_target.toDenseMatrix *:* log(output_layer_state_valid + tiny)) / input.cols

      logger.info(f"$trainingCase: ${title.capitalize} CE $CE%1.3f")
    }

    // TRAIN.
    (1 to epochs).foreach(epoch => {
      logger.info(s"$trainingCase: Epoch $epoch")
      var this_chunk_CE = 0.0


      // LOOP OVER MINI-BATCHES.
      (1 to numbatches).foreach(m => {
        val input_batch = train_input(m - 1)
        val target_batch = train_target(m - 1)

        // FORWARD PROPAGATE.
        // Compute the state of each layer in the network given the input batch
        // and all weights and biases
        val (embedding_layer_state, hidden_layer_state, output_layer_state) =
        fprop(input_batch, word_embedding_weights, embed_to_hid_weights, hid_to_output_weights, hid_bias, output_bias)


        // COMPUTE DERIVATIVE.
        // Expand the target to a sparse 1-of-K vector.
        val expanded_target_batch = expansion_matrix(::, target_batch.toIndexedSequence)
        ///Compute derivative of cross-entropy loss function.
        val error_deriv = (output_layer_state - expanded_target_batch).toDenseMatrix

        // MEASURE LOSS FUNCTION.
        val CE = -sum(expanded_target_batch.toDenseMatrix *:* log(output_layer_state + tiny)) / batchsize



        count = count + 1
        this_chunk_CE = this_chunk_CE + (CE - this_chunk_CE) / count
        trainset_CE = trainset_CE + (CE - trainset_CE) / m

        //        logger.info(f"Batch $m Train CE $this_chunk_CE%1.3f")
        if (m % show_training_CE_after == 0) {
          logger.info(f"$trainingCase: Batch $m Train CE $this_chunk_CE%1.3f")
          count = 0
          this_chunk_CE = 0.0
        }

        // BACK PROPAGATE.
        // OUTPUT LAYER.
        val hid_to_output_weights_gradient = hidden_layer_state * error_deriv.t
        val output_bias_gradient = sum(error_deriv, Axis._1).asDenseMatrix.t
        val back_propagated_deriv_1 = (hid_to_output_weights * error_deriv) *:* hidden_layer_state *:* (1.0 - hidden_layer_state)


        // HIDDEN LAYER.
        // FILL IN CODE. Replace the line below by one of the options.
        // Options:
        // (a) embed_to_hid_weights_gradient = back_propagated_deriv_1' * embedding_layer_state;
        // => (b) embed_to_hid_weights_gradient = embedding_layer_state * back_propagated_deriv_1';
        // (c) embed_to_hid_weights_gradient = back_propagated_deriv_1;
        // (d) embed_to_hid_weights_gradient = embedding_layer_state;
        //        val embed_to_hid_weights_gradient = DenseMatrix.zeros[Double](numhid1 * numwords, numhid2)
        val embed_to_hid_weights_gradient = embedding_layer_state * back_propagated_deriv_1.t


        // FILL IN CODE. Replace the line below by one of the options.
        // Options
        // => (a) hid_bias_gradient = sum(back_propagated_deriv_1, 2);
        // (b) hid_bias_gradient = sum(back_propagated_deriv_1, 1);
        // (c) hid_bias_gradient = back_propagated_deriv_1;
        // (d) hid_bias_gradient = back_propagated_deriv_1';
        //        val hid_bias_gradient = DenseMatrix.zeros[Double](numhid2, 1)
        val hid_bias_gradient = sum(back_propagated_deriv_1, Axis._1).asDenseMatrix.t

        // FILL IN CODE. Replace the line below by one of the options.
        // Options
        // => (a) back_propagated_deriv_2 = embed_to_hid_weights * back_propagated_deriv_1;
        // (b) back_propagated_deriv_2 = back_propagated_deriv_1 * embed_to_hid_weights;
        // (c) back_propagated_deriv_2 = back_propagated_deriv_1' * embed_to_hid_weights;
        // (d) back_propagated_deriv_2 = back_propagated_deriv_1 * embed_to_hid_weights';
        //        val back_propagated_deriv_2 = DenseMatrix.zeros[Double](numhid2, batchsize)
        val back_propagated_deriv_2 = embed_to_hid_weights * back_propagated_deriv_1

        // EMBEDDING LAYER.
        val word_embedding_weights_gradient = (0 until numwords)
          .foldLeft(word_embedding_weights_gradient_init)((wewg, w) => {
            wewg + expansion_matrix(::, input_batch(w, ::).toIndexedSequence()).toDenseMatrix *
              back_propagated_deriv_2(w * numhid1 until (w + 1) * numhid1, ::).t
          })

        //UPDATE WEIGHTS AND BIASES.
        word_embedding_weights_delta = (momentum *:* word_embedding_weights_delta) + (word_embedding_weights_gradient /:/ batchsize.toDouble)
        word_embedding_weights = word_embedding_weights - (learning_rate * word_embedding_weights_delta)

        embed_to_hid_weights_delta = (momentum *:* embed_to_hid_weights_delta) + (embed_to_hid_weights_gradient /:/ batchsize.toDouble)
        embed_to_hid_weights = embed_to_hid_weights - (learning_rate * embed_to_hid_weights_delta)

        hid_to_output_weights_delta = (momentum *:* hid_to_output_weights_delta) + (hid_to_output_weights_gradient /:/ batchsize.toDouble)
        hid_to_output_weights = hid_to_output_weights - (learning_rate * hid_to_output_weights_delta)

        hid_bias_delta = (momentum *:* hid_bias_delta) + (hid_bias_gradient /:/ batchsize.toDouble)
        hid_bias = hid_bias - (learning_rate * hid_bias_delta)

        output_bias_delta = (momentum *:* output_bias_delta) + (output_bias_gradient /:/ batchsize.toDouble)
        output_bias = output_bias - (learning_rate * output_bias_delta)

        // VALIDATE.
        if (m % show_validation_CE_after == 0)
          check("validation", valid_input, valid_target)
      })

      logger.info(f"$trainingCase: Average Training CE $trainset_CE%1.3f")
    })

    logger.info(f"$trainingCase: Final Training CE $trainset_CE%1.3f")

    check("validation", valid_input, valid_target)

    check("test", test_input, test_target)

    val end_time = System.currentTimeMillis()
    val diff = (end_time - start_time) / 1000.0

    logger.info(f"$trainingCase: Training took $diff%.2f seconds")

    NeuralNetworkModel(word_embedding_weights, embed_to_hid_weights, hid_to_output_weights, hid_bias, output_bias, vocabulary)
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
    val batchsize = input_batch.cols


    // COMPUTE STATE OF WORD EMBEDDING LAYER.
    //  Look up the inputs word indices in the word_embedding_weights matrix.
    val embedding_layer_state = word_embedding_weights(
      input_batch
        .reshape(1)
        .toIndexedSequence()
      , ::)
      .toDenseMatrix
      .reshape(numhid1 * numwords)

    // COMPUTE STATE OF HIDDEN LAYER.
    //  Compute inputs to hidden units.
    val inputs_to_hidden_units = (embed_to_hid_weights.t * embedding_layer_state) + tile(hid_bias, 1, batchsize)


    // Apply logistic activation function.
    //  % FILL IN CODE. Replace the line below by one of the options.
    // Options:
    // (a) hidden_layer_state = 1 ./ (1 + exp(inputs_to_hidden_units));
    // (b) hidden_layer_state = 1 ./ (1 - exp(-inputs_to_hidden_units));
    // => (c) hidden_layer_state = 1 ./ (1 + exp(-inputs_to_hidden_units));
    // (d) hidden_layer_state = -1 ./ (1 + exp(-inputs_to_hidden_units));
    //    val hidden_layer_state = DenseMatrix.zeros[Double](numhid2, batchsize)
    val hidden_layer_state = 1.0./(exp(-inputs_to_hidden_units) + 1.0)


    // COMPUTE STATE OF OUTPUT LAYER.
    //  Compute inputs to softmax.
    //  % FILL IN CODE. Replace the line below by one of the options.
    // Options:
    // => (a) inputs_to_softmax = hid_to_output_weights' * hidden_layer_state +  repmat(output_bias, 1, batchsize);
    // (b) inputs_to_softmax = hid_to_output_weights' * hidden_layer_state +  repmat(output_bias, batchsize, 1);
    // (c) inputs_to_softmax = hidden_layer_state * hid_to_output_weights' +  repmat(output_bias, 1, batchsize);
    // (d) inputs_to_softmax = hid_to_output_weights * hidden_layer_state +  repmat(output_bias, batchsize, 1);
    //    val inputs_to_softmax = DenseMatrix.zeros[Double](vocab_size, batchsize)
    val inputs_to_softmax = (hid_to_output_weights.t * hidden_layer_state) + tile(output_bias, 1, batchsize)


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


    assertMatrixDimensions("embedding_layer_state", embedding_layer_state, numhid1 * numwords, batchsize)
    assertMatrixDimensions("hidden_layer_state", hidden_layer_state, numhid2, batchsize)
    assertMatrixDimensions("output_layer_state_norm", output_layer_state_norm, vocab_size, batchsize)

    (embedding_layer_state, hidden_layer_state, output_layer_state_norm)
  }


  def assertMatrixDimensions(name: String, matrix: DenseMatrix[Double], expectedRows: Int, expectedCols: Int): Unit = {
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
  private def restructureInputMatrixBatch(matrix: DenseMatrix[Double]): (List[DenseMatrix[Double]], List[DenseMatrix[Double]]) = {
    val D: Int = matrix.rows - 1
    val N: Int = batchsize
    val M: Int = Math.floor(matrix.cols / N.toDouble).toInt

    (
      hsplit(matrix(0 until D, 0 until (N * M)), M).toList,
      hsplit(matrix(D to D, 0 until (N * M)), M).toList
    )
  }

  /**
    * Splits D * X matrix into D-1 * X matrix and 1 * X matrix. Using last row as target (y)
    *
    * @param matrix DenseMatrix[Double]
    * @return Tuple of input & target matrices
    */
  private def restructureInputMatrix(matrix: DenseMatrix[Double]): (DenseMatrix[Double], DenseMatrix[Double]) = {
    val D: Int = matrix.rows - 1

    (
      matrix(0 until D, ::),
      matrix(D to D, ::)
    )
  }


}
