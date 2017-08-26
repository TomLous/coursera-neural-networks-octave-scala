package util

import breeze.linalg.{DenseMatrix, View}

/**
  * Created by Tom Lous on 26/08/2017.
  * Copyright © 2017 Datlinq B.V..
  */
object DenseMatrixUtils {

  implicit class DenseMatrixImprovements[T](d: DenseMatrix[T]) {

    def reshape(rows: Int, view: View = View.Prefer): DenseMatrix[T] = {
      val cols = d.size / rows
      d.reshape(rows, cols, view)
    }


  }

}
