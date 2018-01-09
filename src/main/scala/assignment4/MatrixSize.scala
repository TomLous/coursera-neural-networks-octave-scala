package assignment4

import breeze.linalg.DenseMatrix

/**
  * Created by Tom Lous on 08/01/2018.
  * Copyright © 2018 Datlinq B.V..
  */
case class MatrixSize(rows: Int, cols: Int) {

  lazy val prod:Int = rows * cols
}

object MatrixSize {
  def apply[V](matrix:DenseMatrix[V]): MatrixSize = MatrixSize(matrix.rows, matrix.cols)
}