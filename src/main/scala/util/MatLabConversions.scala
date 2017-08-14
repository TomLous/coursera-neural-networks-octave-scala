package util

import breeze.linalg.DenseMatrix
import com.jmatio.types.{MLArray, MLDouble, MLNumericArray}

import scala.collection.immutable
import scala.util.Try

/**
  * Created by Tom Lous on 14/08/2017.
  */
object MatLabConversions {

//  implicit def MLArrayAsDenseMatrix(mLArray: MLArray):DenseMatrix = {
//    MLArray
//  }

//  def mlArrayToDenseMatrix[I <: MLNumericArray[A], A, B](mlArray: I)(implicit f: A => B):DenseMatrix[B] = {
////    DenseMatrix(mlArray.getArray: _*)
//
//    DenseMatrix((0 to mlArray.getM).map(m => {
//      val a:immutable.IndexedSeq[B] = (0 to mlArray.getN).map(n => f(mlArray.get(m, n)))
//      a
//    }): _*)
//
//
////
////    val x:Int = for{
////      m <- 0 to mlArray.getM
////      n <- 0 to mlArray.getN
////    } yield mlArray.get(m, n)
////
//  }

  def mlArrayToDenseMatrixDouble(mlArray: MLArray):Option[DenseMatrix[Double]] = {
    Try{
      val m = mlArray.asInstanceOf[MLDouble]
      DenseMatrix(m.getArray: _*)
    }.toOption

  }

  def mlArrayToDenseMatrixDouble2[A <: Number: Manifest](mlArray: MLArray):Option[DenseMatrix[A]] = {
    Try{
      val mlArray2:MLNumericArray[A] = mlArray.asInstanceOf[MLNumericArray[A]]

      val l:immutable.IndexedSeq[immutable.IndexedSeq[A]] = (0 until mlArray2.getM).map(
        m => {
          val a:immutable.IndexedSeq[A] =
            (0 until mlArray2.getN)
              .map(n => mlArray2.get(m, n))
          a
        })
      DenseMatrix(l: _*)
    }.toOption
  }

}
