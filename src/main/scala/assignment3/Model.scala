package assignment3

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.cos

/**
  * Created by Tom Lous on 10/10/2017.
  * Copyright © 2017 Datlinq B.V..
  */
case class Model(numberHiddenUnits: Int, inputToHidden: DenseMatrix[Double], hiddenToClassification:DenseMatrix[Double]) {


  /**
    * This function takes a model (or gradient in model form), and turns it into one long vector. See also theta_to_model.
    * org: function ret = model_to_theta(model)
    */
  lazy val theta:Theta = {
    val input_to_hid_transpose = inputToHidden.t
    val hid_to_class_transpose = hiddenToClassification.t

    Theta(DenseVector.vertcat(input_to_hid_transpose.toDenseVector, hid_to_class_transpose.toDenseVector), Model.NUM_INPUT_UNITS, Model.NUM_OUTPUT_UNITS)



  }

}

object Model{

  val NUM_INPUT_UNITS:Int = 256
  val NUM_OUTPUT_UNITS:Int = 10 // digits

  /**
    * org: function ret = initial_model(n_hid)
    * @param numberHiddenUnits amount of hidden units
    * @return Model
    */
  def apply(numberHiddenUnits: Int): Model = {
    // n_params = (256+10) * n_hid;
    val numberOfParams = (NUM_INPUT_UNITS+NUM_OUTPUT_UNITS) * numberHiddenUnits

    // as_row_vector = cos(0:(n_params-1));
    val asRowVector = cos(DenseVector[Double](
      (0 until numberOfParams)
        .map(_.toDouble):_*)
    )

    // ret = theta_to_model(as_row_vector(:) * 0.1); % We don't use random initialization, for this assignment. This way, everybody will get the same results.
    Theta(asRowVector * 0.1, NUM_INPUT_UNITS, NUM_OUTPUT_UNITS).model
  }
}
