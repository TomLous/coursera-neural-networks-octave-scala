package assignment3

import org.scalatest.FunSuite

/**
  * Created by Tom Lous on 10/10/2017.
  * Copyright © 2017 Datlinq B.V..
  */
class Assignment3Test extends FunSuite {

  test("Model structure") {
    (3 to 10).foreach {n =>

      val m = Model(n)

      assert(m.inputToHidden.cols === n, "i->h cols")
      assert(m.inputToHidden.rows === Model.NUM_INPUT_UNITS, "i->h rows")

      assert(m.hiddenToClassification.cols === n, "h->c cols")
      assert(m.hiddenToClassification.rows === Model.NUM_OUTPUT_UNITS, "h->c rows")

      assert(m.numberHiddenUnits === n, "hidden units")
    }

  }

}
