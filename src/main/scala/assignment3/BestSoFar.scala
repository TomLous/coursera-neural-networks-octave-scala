package assignment3

/**
  * Created by Tom Lous on 08/12/2017.
  * Copyright © 2017 Datlinq B.V..
  */
case class BestSoFar(
                      theta: Theta, // this will be overwritten soon
                      validationLoss: Double = Double.PositiveInfinity,
                      afterNIteratons: Int = -1
                    )
