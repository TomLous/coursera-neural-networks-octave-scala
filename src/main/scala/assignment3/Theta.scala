package assignment3

import breeze.linalg.DenseVector

/**
  * Created by Tom Lous on 10/10/2017.
  */
case class Theta(thetaVector: DenseVector[Double], numberInputUnits:Int = 256, numberOutputUnits: Int = 10) {


  /**
    * This value takes a model (or gradient) in the form of one long vector (maybe produced by model_to_theta), and restores it to the structure format, i.e. with fields .input_to_hid and .hid_to_class, both matrices.
    * org: function ret = theta_to_model(theta)
    */
  lazy val model:Model = {
    // n_hid = size(theta, 1) / (256+10);
    val numberHiddenUnits = thetaVector.length / (numberInputUnits + numberOutputUnits)

    // ret.input_to_hid = transpose(reshape(theta(1: 256*n_hid), 256, n_hid));
    val inputToHidden = thetaVector(0 until numberHiddenUnits * numberInputUnits)
      .toDenseMatrix
      .reshape(numberInputUnits, numberHiddenUnits).t.toDenseMatrix

    //  ret.hid_to_class = reshape(theta(256 * n_hid + 1 : size(theta,1)), n_hid, 10).';
    val hiddenToClassification = thetaVector(numberHiddenUnits * numberInputUnits to -1)
      .toDenseMatrix
      .reshape(numberHiddenUnits, numberOutputUnits)
      .t
      .toDenseMatrix


    Model(numberHiddenUnits, inputToHidden, hiddenToClassification)
  }



}
