package assignment4

import assignment3.DataBundle
import breeze.linalg.DenseMatrix

/**
  * Created by Tom Lous on 09/01/2018.
  * Copyright © 2018 Datlinq B.V..
  */
trait GradientFunction {
  def run(rbmWeights:DenseMatrix[Double], visibleData:DataBundle):DenseMatrix[Double]

}


object CD1 extends GradientFunction {
  override def run(rbmWeights: DenseMatrix[Double], visibleData: DataBundle): DenseMatrix[Double] = ???
}

object ClassificationPhiGradient extends GradientFunction {
  override def run(rbmWeights: DenseMatrix[Double], visibleData: DataBundle): DenseMatrix[Double] = ???
}