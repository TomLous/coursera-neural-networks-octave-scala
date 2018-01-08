package assignment4

import breeze.linalg._
import com.typesafe.scalalogging.LazyLogging

import scala.math._

/**
  * Created by Tom Lous on 08/01/2018.
  * Copyright © 2018 Datlinq B.V..
  */
case class RandomDataSource(randomSource: DenseVector[Double]) extends LazyLogging{


  /**
    * a4_rand(requested_size, seed)
    * @param requestedSize Requested Matrix Size
    * @param seed randomizer seed
    * @return slice out of RandomSource Vector based on requestedSize
    */
  def rand(requestedSize:MatrixSize, seed: Double):DenseMatrix[Double] = {
    val startI = round(seed).toInt % round(randomSource.length / 10.0).toInt // start_i = mod(round(seed), round(size(randomness_source, 2) / 10)) + 1;

    if(startI + requestedSize.prod > randomSource.length){ // if start_i + prod(requested_size) >= size(randomness_source, 2) + 1,
      logger.error("a4_rand failed to generate an array of that size (too big)")
    }

    randomSource.slice(startI, startI + requestedSize.prod).toDenseMatrix.reshape(requestedSize.rows, requestedSize.cols) //ret = reshape(randomness_source(start_i : start_i+prod(requested_size)-1), requested_size);
  }

}
