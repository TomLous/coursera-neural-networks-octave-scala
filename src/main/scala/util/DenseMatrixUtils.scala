package util

import breeze.linalg.{DenseMatrix, DenseVector, Transpose, View}

import scala.reflect.ClassTag

/**
  * Created by Tom Lous on 26/08/2017.
  */
object DenseMatrixUtils {

  implicit class DenseMatrixImprovements[T](d: DenseMatrix[T]) {

    def reshape(rows: Int, view: View = View.Prefer): DenseMatrix[T] = {
      val cols = d.size / rows
      d.reshape(rows, cols, view)
    }

    def toIndexedSequence():IndexedSeq[Int] = {
      d.asInstanceOf[DenseMatrix[Double]]
        .toArray
        .map(_.toInt - 1) // indices based on matlab => -1
        .toIndexedSeq
    }


  }

  implicit class TransposeDenseMatrixImprovements[T: ClassTag](t: Transpose[DenseMatrix[T]]) {
    def toIndexedSequence():IndexedSeq[Int] = {
      t.t
//        .toDenseMatrix
        .asInstanceOf[DenseMatrix[Double]]
        .toArray
        .map(_.toInt - 1) // indices based on matlab => -1
        .toIndexedSeq
    }


  }

  implicit class TransposeDenseVectorImprovements[T: ClassTag](t: Transpose[DenseVector[T]]) {
    def toIndexedSequence():IndexedSeq[Int] = {
      t.t
        .toDenseVector
        .asInstanceOf[DenseVector[Double]]
        .toArray
        .map(_.toInt - 1) // indices based on matlab => -1
        .toIndexedSeq
    }


  }

}
