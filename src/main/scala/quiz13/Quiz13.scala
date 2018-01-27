package quiz13

import com.typesafe.scalalogging.LazyLogging

case class SBNConfig(h1: Int, h2: Int, w1: Double, w2: Double, v: Int){
  lazy val probabiltyPrior:Double = SBNConfig.probability(v * h1 * w1 + v * h2 * w2)

  private def derivative(x:Int) = x * (v - probabiltyPrior)
  lazy val derivativeH1:Double = derivative(h1)
  lazy val derivativeH2:Double = derivative(h2)

}

object SBNConfig{
  def probability(energy: Double):Double = 1 / (1 + math.exp(-energy))

  def bayes(P0:Double, config1: SBNConfig, config2: SBNConfig):Double = {
    val P1 = config1.probabiltyPrior
    val P2 = config2.probabiltyPrior

    P1 * P0 / (P1 * P0 + P2 * ( 1 - P0))
  }
}

/**
  * Created by Tom Lous on 26/01/2018.
  * Copyright © 2018 Datlinq B.V..
  */
object Quiz13 extends App with LazyLogging {


  val Ph1_1 = 0.5 // on or off P(h1=1)
  val Ph2_1 = 0.5 // on or off P(h2=1)

  val C011a = SBNConfig(h1 = 0, h2 = 1, w1 = -6.90675478, w2 = 0.40546511, v = 1)

  logger.info("Q2. P(v=1|h1=0,h2=1)? " + C011a.probabiltyPrior)

  logger.info("Q3. P(h1=0,h2=1, v=1)? " + C011a.probabiltyPrior * (1 - Ph1_1) * Ph2_1)

  logger.info("Q4. ∂logP(C011a)/∂w1? " + C011a.derivativeH1)
  logger.info("Q5. ∂logP(C011a)/∂w2? " + C011a.derivativeH2)

  val C011b = SBNConfig(h1 = 0, h2 = 1, w1 = 10.0, w2 = -4.0, v = 1)
  val C001b = SBNConfig(h1 = 0, h2 = 0, w1 = 10.0, w2 = -4.0, v = 1)


  logger.info("Q6. P(h2=1|v=1,h1=0) = P(v=1|h1=0,h2=1)P(h2=1) / (P(v=1|h1=0,h2=1)P(h2=1) + P(v=1|h1=0,h2=0)P(h2=0))? " + SBNConfig.bayes(Ph2_1, C011b, C001b))

  val C111b = SBNConfig(h1 = 1, h2 = 1, w1 = 10.0, w2 = -4.0, v = 1)
  val C101b = SBNConfig(h1 = 1, h2 = 0, w1 = 10.0, w2 = -4.0, v = 1)

  logger.info("Q7. P(h2=1|v=1,h1=0) = P(v=1|h1=0,h2=1)P(h2=1) / (P(v=1|h1=0,h2=1)P(h2=1) + P(v=1|h1=0,h2=0)P(h2=0))? " +  SBNConfig.bayes(Ph2_1, C111b, C101b))


}
