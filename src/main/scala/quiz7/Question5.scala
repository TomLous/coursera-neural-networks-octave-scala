package quiz7

import com.typesafe.scalalogging.LazyLogging

/**
  * Created by Tom Lous on 01/10/2017.
  * Copyright © 2017 Datlinq B.V..
  */
object Question5 extends App with LazyLogging{


  val case3 = RNN(
    w_xh = 1.0,
    w_hh = -2.0,
    w_hy = 1.0,
    h_bias = 0.0,
    y_bias = 0.0,
    x_inputs = List(1.0,1.0, 1.0, 1.0),
    t_targets = List(0.5,0.5,0.5, 0.5)
  )


  println(case3)


}
