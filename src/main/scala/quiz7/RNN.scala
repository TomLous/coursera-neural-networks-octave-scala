package quiz7

import com.typesafe.scalalogging.LazyLogging

import scala.math.exp

/**
  * Created by Tom Lous on 01/10/2017.
  * Copyright © 2017 Datlinq B.V..
  */
case class RNN(
                w_xh: Double,
                w_hh: Double,
                w_hy: Double,
                h_bias: Double,
                y_bias: Double,
                x_inputs: List[Double]
              ) extends LazyLogging {

  def logisticActivation(k: Double):Double = 1.0 / ( exp(-k) + 1.0 )
  def logit(x: Double, h:Double):Double = logisticActivation(x * w_xh + h * w_hh + h_bias)

  lazy val (final_h,y_res) = x_inputs
    .zipWithIndex
    .foldLeft((0.0, List.empty[(Double, Int)])){
      case ((h_prev, y_list), (x_input, timestep)) => {
        logger.debug(s"Timestep $timestep: Input: $x_input, Prev h: $h_prev")

        val h_output = logit(x_input, h_prev)
        val y_output =  h_output * w_hy + y_bias
        (h_output, (y_output, timestep) :: y_list)
      }}


  override def toString: String = {
    val rnn = s"RNN(w_xh: $w_xh, w_hh: $w_hh, w_hy: $w_hy, h_bias: $h_bias, y_bias: $y_bias, x_inputs: $x_inputs)"
    s"$rnn: \n" +
      s"last h: $final_h" +
      s"\ny_output: \n" + y_res.reverse.map(y => s"y_${y._2}: ${y._1} ").mkString("\n")


  }
}

