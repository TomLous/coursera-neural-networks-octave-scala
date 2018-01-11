package assignment4

/**
  * Created by Tom Lous on 11/01/2018.
  * Copyright © 2018 Datlinq B.V..
  */
object Utils {
  def ~=(x: Double, y: Double, precision: Double) = {
     if ((x - y).abs < precision) true else false
  }

  def assertDouble(a: Double, b:Double) = {
    assert(~=(a, b, 0.00001), s"$a != $b")
  }

}
