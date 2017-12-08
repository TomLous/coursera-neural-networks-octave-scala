package assignment3

import breeze.linalg.DenseMatrix

/**
  * Created by Tom Lous on 13/10/2017.
  * Copyright © 2017 Datlinq B.V..
  */
case class DataBundle(inputs: DenseMatrix[Double], targets: DenseMatrix[Double]) {
   def batch(offset: Int, size: Int): DataBundle = {
     DataBundle(
       inputs(::, offset until offset + size), //datas.training.inputs(:, training_batch_start : training_batch_start + mini_batch_size - 1);
        targets(::, offset until offset + size) //datas.training.inputs(:, training_batch_start : training_batch_start + mini_batch_size - 1);
     )
   }
}
object DataBundle{

  def apply(inputsOpt: Option[DenseMatrix[Double]], targetsOpt: Option[DenseMatrix[Double]]): DataBundle = {
    assert(inputsOpt.isDefined)
    assert(targetsOpt.isDefined)
    new DataBundle(inputsOpt.get, targetsOpt.get)
  }
}
