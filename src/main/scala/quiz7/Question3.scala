package quiz7

//import breeze.linalg._
//import breeze.numerics._
//import com.typesafe.scalalogging.Logger
//import util.DenseMatrixUtils._
import com.typesafe.scalalogging.LazyLogging

/**
  * Created by Tom Lous on 01/10/2017.
  * The network parameters are: Wxh=0.5 , Whh=−1.0 , Why=−0.7 , hbias=−1.0, and ybias=0.0. Remember, σ(k)=1/1+exp(−k).
  *  If the input x takes the values 9,4,−2 at time steps 0,1,2 respectively,
  *  what is the value of the hidden state h at T=2?
  *  Give your answer with at least two digits after the decimal point.
  */
object Question3 extends App with LazyLogging{


 val case1 = RNN(
   w_xh = 0.5,
   w_hh = -1.0,
   w_hy = -0.7,
   h_bias = -0.7,
   y_bias = 0.0,
   x_inputs = List(9.0, 4.0, -2.0)
 )

  val case2 = RNN(
    w_xh = -0.1,
    w_hh = 0.5,
    w_hy = 0.25,
    h_bias = 0.4,
    y_bias = 0.0,
    x_inputs = List(18.0,9.0, -8.0)
  )


  println(case1)
  println(case2)







}
