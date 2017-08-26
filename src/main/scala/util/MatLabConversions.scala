package util

import breeze.linalg.DenseMatrix
import com.jmatio.types._

import scala.util.Try

/**
  * Created by Tom Lous on 14/08/2017.
  */
object MatLabConversions {


  /**
    * Converts a MLArray of java.lang.Double (or any Number) to a DenseMatrix[Double]
    * @todo Make the returntype dependant on input type (java.lang.Double => Double, java.lang.Integer => Int)
    * @param mlArray MLArray
    * @tparam A (any Java number)
    * @return Option of a DenseMatrix (now always Double)
    */
  implicit def mlArrayToDenseMatrix[A <: Number](mlArray: MLArray): Option[DenseMatrix[Double]] = {
    Try {
      val mlNumericArray: MLNumericArray[A] = mlArray.asInstanceOf[MLNumericArray[A]]

      def convert(a: A):Double = a.doubleValue() // @todo make this implicit based on type

      DenseMatrix(
        (0 until mlNumericArray.getM)
          .map(m => (0 until mlNumericArray.getN)
            .map(n => convert(mlNumericArray.get(m, n)))
          ): _*
      )

    }.toOption
  }

  /**
    * Converts a MLArray as MLCell of char arrays to a List[Strings]
    * @param mlArray MLArray
    * @return Option of a List of Strings
    */
  implicit def mlArrayToStringList(mlArray: MLArray): Option[List[String]] = {
    Try {
      val mlCell: MLCell = mlArray.asInstanceOf[MLCell]

      def convert(a: MLChar):String = a.getString(0)

      (0 until mlCell.getM)
        .flatMap(m => (0 until mlCell.getN)
          .map(n => convert(mlCell.get(m, n).asInstanceOf[MLChar]))
        ).toList


    }.toOption
  }


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

  //  def mlArrayToDenseMatrixDouble(mlArray: MLArray): Option[DenseMatrix[Double]] = {
  //    Try {
  //      val m = mlArray.asInstanceOf[MLDouble]
  //      DenseMatrix(m.getArray: _*)
  //    }.toOption
  //
  //  }

}
