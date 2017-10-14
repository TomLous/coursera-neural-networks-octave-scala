package assignment3

import breeze.linalg.DenseMatrix

/**
  * Created by Tom Lous on 13/10/2017.
  * Copyright © 2017 Datlinq B.V..
  */
case class DataBundle(inputs: DenseMatrix[Double], targets: DenseMatrix[Double])
object DataBundle{

  def apply(inputsOpt: Option[DenseMatrix[Double]], targetsOpt: Option[DenseMatrix[Double]]): DataBundle = {
    assert(inputsOpt.isDefined)
    assert(targetsOpt.isDefined)
    new DataBundle(inputsOpt.get, targetsOpt.get)
  }
}
