package quiz7


import com.typesafe.scalalogging.LazyLogging

import scala.math._

/**
  * Created by Tom Lous on 01/10/2017.
  */
case class RNN(
                w_xh: Double,
                w_hh: Double,
                w_hy: Double,
                h_bias: Double,
                y_bias: Double,
                x_inputs: List[Double],
                t_targets: List[Double] = Nil
              ) extends LazyLogging {

  def logisticActivation(k: Double):Double = 1.0 / ( exp(-k) + 1.0 )


  lazy val (final_h,y_res,error) = {





    x_inputs
      .zipAll(t_targets, 0.0, 0.0)
      .zipWithIndex
      .map{case ((a,b), c) => (a,b,c)}
      .foldLeft((0.0, List.empty[(Double, Int)], 0.0)){
        case ((h_prev, y_list, e_total), (x_input, t_input, timestep)) => {

          logger.debug(s"Timestep $timestep: Input: $x_input, Prev h: $h_prev, t: $t_input")

          val z_output =  w_xh * x_input + w_hh * h_prev + h_bias
          val h_output = logisticActivation(z_output)
          val y_output =  w_hy * h_output + y_bias
          val e_output = 0.5 * pow(t_input - y_output,2)

          val d_e_output = t_input - y_output

          val e_new = e_total + e_output

//          val eder1 = (e_new / y_output) * (y_output / h_output) * (h_output / z_output)
//          val eder2 = (e_output / y_output) * (y_output / h_output) * (h_output / z_output)

          logger.debug(s"Timestep $timestep: Output: z: $z_output, h: $h_output, y: $y_output, e: $e_output")


          (h_output, (y_output, timestep) :: y_list, e_new)
        }}
  }


  override def toString: String = {
    val rnn = s"RNN(w_xh: $w_xh, w_hh: $w_hh, w_hy: $w_hy, h_bias: $h_bias, y_bias: $y_bias, x_inputs: $x_inputs, t_targets: $t_targets)"
    s"$rnn: \n" +
      s"last h: $final_h" +
      s"\ny_output: \n" + y_res.reverse.map(y => s"y_${y._2}: ${y._1} ").mkString("\n")+
      s"\nE: $error"

  }
}

