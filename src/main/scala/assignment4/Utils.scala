package assignment4

/**
  * Created by Tom Lous on 11/01/2018.
  * Copyright © 2018 Datlinq B.V..
  */
object Utils {
  def ~=(x: Double, y: Double, precision: Double) = {
     if ((x - y).abs < precision) true else false
  }

  def assertDouble(a: Double, b:Double, precision:Double=0.00001) = {
    assert(~=(a, b, precision), s"$a != $b")
  }

  def assertDimensions(a:MatrixSize, b:MatrixSize) = {
    assert(a.cols == b.cols, s"${a.cols} != ${a.cols}")
    assert(a.rows == b.rows, s"${b.rows} != ${b.rows}")
  }

}
