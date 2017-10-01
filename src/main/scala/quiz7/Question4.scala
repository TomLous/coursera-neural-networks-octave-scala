package quiz7

import com.typesafe.scalalogging.LazyLogging

/**
  * Created by Tom Lous on 01/10/2017.
  * The network parameters are: Wxh=0.5 , Whh=−1.0 , Why=−0.7 , hbias=−1.0, and ybias=0.0. Remember, σ(k)=1/1+exp(−k).
  *  If the input x takes the values 9,4,−2 at time steps 0,1,2 respectively,
  *  what is the value of the hidden state h at T=2?
  *  Give your answer with at least two digits after the decimal point.
  */
object Question4 extends App with LazyLogging{


  val case2 = RNN(
    w_xh = -0.1,
    w_hh = 0.5,
    w_hy = 0.25,
    h_bias = 0.4,
    y_bias = 0.0,
    x_inputs = List(18.0,9.0, -8.0),
    t_targets = List(0.1,0.1,0.2)
  )

//  If the target output values are t0=0.1,t1=−0.1,t2=−0.2 and the squared error loss is used, what is the value of the error derivative just before the hidden unit nonlinearity at T=2 (i.e. ∂E∂z2)? Write your answer up to at least the fourth decimal place.
//  0.0159

//  If the target output values are t0=0.1,t1=−0.1,t2=−0.2 and the squared error loss is used, what is the value of the error derivative just before the hidden unit nonlinearity at T=2 (i.e. ∂E∂z2)? Write your answer up to at least the fourth decimal place.
//  0.020053



  println(case2)






}
