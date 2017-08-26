package assignment1

import breeze.linalg._
import breeze.plot._
import com.typesafe.scalalogging.LazyLogging

import scala.annotation.tailrec

/**
  * Learns the weights of a perceptron for a 2-dimensional dataset and plots
  * the perceptron at each iteration where an iteration is defined as one
  * full pass through the data. If a generously feasible weight vector
  * is provided then the visualization will also show the distance
  * of the learned weight vectors to the generously feasible weight vector.
  *
  * @param name                Name of the perceptron
  * @param neg_examples_nobias The num_neg_examples x 2 matrix for the examples with target 0.
  * @param pos_examples_nobias The num_pos_examples x 2 matrix for the examples with target 1.
  * @param w_init              A 3-dimensional initial weight vector. The last element is the bias.
  * @param w_gen_feas          he learned weight vector.
  * @param learning_rate       learning rate of the perceptron
  */
case class Perceptron(
                       name: String,
                       neg_examples_nobias: DenseMatrix[Double],
                       pos_examples_nobias: DenseMatrix[Double],
                       w_init: Option[DenseVector[Double]],
                       w_gen_feas: Option[DenseVector[Double]],
                       learning_rate: Double = 1.0
                     ) extends LazyLogging {

  val maxIter = 30

  // Bookkeeping
  val num_neg_examples: Int = neg_examples_nobias.rows
  val num_pos_examples: Int = pos_examples_nobias.rows

  private[this] var num_err_history: List[Int] = Nil
  private[this] var w_dist_history: List[Double] = Nil

  // Here we add a column of ones to the examples in order to allow us to lear bias parameters.
  val neg_examples: DenseMatrix[Double] = DenseMatrix.horzcat(neg_examples_nobias, DenseMatrix.ones[Double](num_neg_examples, 1))
  val pos_examples: DenseMatrix[Double] = DenseMatrix.horzcat(pos_examples_nobias, DenseMatrix.ones[Double](num_pos_examples, 1))

  // If weight vectors have not been provided, initialize them appropriately.
  val w0: DenseVector[Double] = w_init match {
    case Some(wi) => wi
    case None => DenseVector.rand[Double](3)
  }

  private[this] var w: DenseVector[Double] = w0

  /**
    * Learns the weights of a perceptron for a 2-dimensional dataset
    *
    * @return new weightVector
    */
  def learn(): Either[String, DenseVector[Double]] = {

    val figure = Figure(name)

    /**
      * Recusivily iterate until no errors are found
      *
      * @param iteration       counter
      * @param w               current weights
      * @param num_err_history history
      * @param w_dist_history  history
      * @return Either an error message or the calculated wieght + history)
      */
    @tailrec
    def iterate(iteration: Int, w: DenseVector[Double], num_err_history: List[Int], w_dist_history: List[Double]): Either[String, (DenseVector[Double], List[Int], List[Double])] = {
      if (iteration >= maxIter) Left(s"Number of iterations exeeded without converging: $maxIter") // kill the loop if not converging
      else if (num_err_history.head == 0) {
        evaluate(w, iteration, num_err_history, w_dist_history)
        println(s"Converged $name after ${iteration - 1} steps")
        Right((w, num_err_history.reverse, w_dist_history.reverse))
      } // no more errors => break
      else {
        // update weights
        val w_updated = update_weights(w)

        // evaluate new weights
        val (num_err_history_updated, w_dist_history_updated) = evaluate(w_updated, iteration, num_err_history, w_dist_history)

        // tailrec
        iterate(iteration + 1, w_updated, num_err_history_updated, w_dist_history_updated)
      }
    }

    /**
      * Calculate the mistakes when applying new weights to samples
      *
      * @param w               new weights
      * @param iter            loop counter
      * @param num_err_history current history
      * @param w_dist_history  current history
      * @return new histories
      */
    def evaluate(w: DenseVector[Double], iter: Int, num_err_history: List[Int], w_dist_history: List[Double]): (List[Int], List[Double]) = {
      // positive & negative mistakes
      val (mistakes0, mistakes1) = eval(w)

      // count errors
      val num_errs = mistakes0.length + mistakes1.length

      logger.info(s"""Number of errors in iteration $iter: $num_errs""")
      logger.info(s"""\tWeights $w""")

      // calculate weight distance
      val w_dist = w_gen_feas.map(wgf => {
        norm(wgf - w)
      })

      // update histories
      val w_dist_history_updated = w_dist.foldLeft(w_dist_history)((hist, d) => d :: hist)
      val num_err_history_updated = num_errs :: num_err_history

      // plot the data
      plot_perceptron(figure, s"$name-$iter", mistakes0, mistakes1, num_err_history, w, w_dist_history_updated)

      // return histories
      (num_err_history_updated, w_dist_history_updated)
    }

    /**
      * Non functional method to update some vals. Can be removed, but kept for historic perspective?
      *
      * @param new_w               w := new_w
      * @param new_num_err_history num_err_history := new_num_err_history
      * @param new_w_dist_history  w_dist_history := new_w_dist_history
      */
    def updateVars(new_w: DenseVector[Double], new_num_err_history: List[Int], new_w_dist_history: List[Double]): Unit = {
      w = new_w
      num_err_history = new_num_err_history
      w_dist_history = new_w_dist_history
    }

    def savePlot() = figure.saveas(s"src/main/resources/assignment1/plots/image-$name.png")

    // initial evaluation (iteration 0)
    val (num_err_history0, w_dist_history0) = evaluate(w0, 0, Nil, Nil)

    // iterate until converged or some error
    iterate(1, w0, num_err_history0, w_dist_history0) match {
      case Right((w_updated, num_err_history_updated, w_dist_history_updated)) => {
        updateVars(w_updated, num_err_history_updated, w_dist_history_updated)
        savePlot
        Right(w_updated)
      }
      case Left(x) => {
        savePlot
        Left(x)
      }
    }
  }


  /**
    * Evaluates the perceptron using a given weight vector. Here, evaluation
    * refers to finding the data points that the perceptron incorrectly classifies.
    * Inputs:
    * neg_examples - The num_neg_examples x 3 matrix for the examples with target 0.
    * pos_examples- The num_pos_examples x 3 matrix for the examples with target 1.
    *
    * @param w A 3-dimensional weight vector, the last element is the bias.
    * @return tuple of DenseVectors
    *         mistakes0 - A vector containing the indices of the negative examples that have been
    *         incorrectly classified as positive.
    *         mistakes0 - A vector containing the indices of the positive examples that have been
    *         incorrectly classified as negative.
    */
  def eval(w: DenseVector[Double]): (List[Int], List[Int]) = {
    (findMistakes(neg_examples, w, t => t._1 > 0),
      findMistakes(pos_examples, w, t => t._1 < 0))
  }

  /**
    * Foreach tuple in matrix, calculate the activation (tuple' * weight) and mark index as mistake if not filtered away by method
    *
    * @param samples          DenseMatrix of samples
    * @param w                weight
    * @param filterActivation filter method that takes activationValue & index
    * @return list of indices in samples that are wrong
    */
  private def findMistakes(samples: DenseMatrix[Double], w: DenseVector[Double], filterActivation: ((Double, Int)) => Boolean): List[Int] = {
    samples(*, ::)
      .map(x => x.t * w)
      .toScalaVector()
      .zipWithIndex
      .filter(filterActivation)
      .map(_._2)
      .toList
  }


  /**
    * Updates the weights of the perceptron for incorrectly classified points
    * using the perceptron update algorithm. This function makes one sweep
    * over the dataset.
    * Inputs:
    * neg_examples - The num_neg_examples x 3 matrix for the examples with target 0.
    * pos_examples- The num_pos_examples x 3 matrix for the examples with target 1.
    *
    * @param w A 3-dimensional weight vector, the last element is the bias.
    * @return DenseVector The weight vector after one pass through the dataset using the perceptron learning rule.
    */
  def update_weights(w: DenseVector[Double]): DenseVector[Double] = {
    val w1 = neg_examples(*, ::)
      .foldLeft(w)((current_w, this_case) => {
        val activation = this_case.t * current_w
        if (activation >= 0) {
          current_w + learning_rate * -this_case
        }
        else {
          current_w
        }
      })

    val w2 = pos_examples(*, ::)
      .foldLeft(w1)((current_w, this_case) => {
        val activation = this_case.t * current_w
        if (activation < 0) {
          current_w + learning_rate * this_case
        }
        else {
          current_w
        }
      })

    w2
  }


  /**
    * Plots information about a perceptron classifier on a 2-dimensional dataset.
    * The top-left plot shows the dataset and the classification boundary given by
    * the weights of the perceptron. The negative examples are shown as dots
    * while the positive examples are shown as pluses. If an example is colored
    * green then it means that the example has been correctly classified by the
    * provided weights. If it is colored red then it has been incorrectly classified.
    * The top-right plot shows the number of mistakes the perceptron algorithm has
    * made in each iteration so far.
    * The bottom-left plot shows the distance to some generously feasible weight
    * vector if one has been provided (note, there can be an infinite number of these).
    * Points that the classifier has made a mistake on are shown in red,
    * while points that are correctly classified are shown in green.
    * The goal is for all of the points to be green (if it is possible to do so).
    * Inputs:
    * neg_examples - The num_neg_examples x 3 matrix for the examples with target 0.
    * pos_examples- The num_pos_examples x 3 matrix for the examples with target 1.
    *
    * @param mistakes0       A vector containing the indices of the datapoints from class 0 incorrectly classified by the perceptron. This is a subset of neg_examples.
    * @param mistakes1       A vector containing the indices of the datapoints from class 1 incorrectly classified by the perceptron. This is a subset of pos_examples.
    * @param num_err_history A vector containing the number of mistakes for each iteration of learning so far.
    * @param w               A 3-dimensional vector corresponding to the current weights of the perceptron. The last element is the bias.
    * @param w_dist_history  A vector containing the L2-distance to a generously feasible weight vector for each iteration of learning so far. Empty if one has not been provided.
    */
  def plot_perceptron(figure: Figure, name: String, mistakes0: List[Int], mistakes1: List[Int], num_err_history: List[Int], w: DenseVector[Double], w_dist_history: List[Double]): Unit = {

    figure.clear()

    val neg_correct_ind = (0 until num_neg_examples).diff(mistakes0)
    val pos_correct_ind = (0 until num_pos_examples).diff(mistakes1)

    val p0 = figure.subplot(2, 2, 0)
    p0.title = "Classifier"

    p0 += plot(neg_examples(neg_correct_ind, 0).toArray, neg_examples(neg_correct_ind, 1).toArray, '.', colorcode = "[43,146,31]")
    p0 += plot(pos_examples(pos_correct_ind, 0).toArray, pos_examples(pos_correct_ind, 1).toArray, '+', colorcode = "[43,146,31]")
    p0 += plot(neg_examples(mistakes0, 0).toArray, neg_examples(mistakes0, 1).toArray, '.', colorcode = "[220,0,25]")
    p0 += plot(pos_examples(mistakes1, 0).toArray, pos_examples(mistakes1, 1).toArray, '+', colorcode = "[220,0,25]")

    val bound = 5.0
    p0 += plot(List(-bound, bound), List((-w(2) + bound * w(0)) / w(1), (-w(2) - bound * w(0)) / w(1)))

    p0.xlim = (-1.0, 1.0)
    p0.ylim = (-1.0, 1.0)


    val p1 = figure.subplot(1)
    p1.title = "Number of errors"
    //    p1.xlabel = "iteration"
    //    p1.ylabel = "errors"
    p1.ylim = (0.0, num_pos_examples + num_neg_examples)
    p1 += plot(num_err_history.indices, num_err_history.reverse)

    if (w_dist_history.nonEmpty) {
      val p2 = figure.subplot(2)
      p2.title = "Distance"
      //    p2.xlabel = "iteration"
      //    p2.ylabel = "distance"
      p2.ylim = (0.0, 15.0)
      p2 += plot(w_dist_history.indices.map(_.toDouble), w_dist_history.reverse)
    }

    //    figure.saveas(s"target/image-$name.png")

  }



}
