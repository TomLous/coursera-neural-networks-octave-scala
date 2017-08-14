package util

import breeze.linalg.DenseMatrix
import com.jmatio.types.{MLArray, MLDouble, MLNumericArray}

import scala.collection.immutable
import scala.util.Try
import collection.JavaConverters._
import scala.reflect.ClassTag

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

  def mlArrayToDenseMatrixDouble(mlArray: MLArray): Option[DenseMatrix[Double]] = {
    Try {
      val m = mlArray.asInstanceOf[MLDouble]
      DenseMatrix(m.getArray: _*)
    }.toOption

  }

  def mlArrayToDenseMatrix[A <: Number](mlArray: MLArray): Option[DenseMatrix[Double]] = {
    Try {
      val mlNumericArray: MLNumericArray[A] = mlArray.asInstanceOf[MLNumericArray[A]]

      def convert(a: A) = a.doubleValue()

      DenseMatrix(
        (0 until mlNumericArray.getM)
          .map(m => (0 until mlNumericArray.getN)
            .map(n => convert(mlNumericArray.get(m, n)))
          ): _*
      )

    }.toOption
  }

}
