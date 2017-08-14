package util

import io.MatLabFile
import org.scalatest.{BeforeAndAfterEach, FunSuite, fixture}

/**
  * Created by Tom Lous on 14/08/2017.
  * Copyright © 2017 Datlinq B.V..
  */
class MatLabConversionsTest extends fixture.FunSuite {


  type FixtureParam = MatLabFile

  def withFixture(test: OneArgTest) = {
    val inputFileName = "/dataset1_test.mat"
    val inputFilePath = getClass.getResource(inputFileName).toURI
    val mlFile = MatLabFile(inputFilePath)
    test(mlFile)
  }

  test("testMlArrayToDenseMatrix") { mlFile => {
      val mlArray = mlFile.mlArrayOption("neg_examples_nobias")
      val double = MatLabConversions.mlArrayToDenseMatrixDouble(mlArray.get)

    println(double)


    }
  }

//  test("testMlArrayToDenseMatrix") { mlFile => {
//    val mlArray = mlFile.mlArrayOption("neg_examples_nobias")
//    val double = MatLabConversions.mlArrayToDenseMatrixDouble2(mlArray.get)
//
//    println(double)
//
//
//  }
//  }

}
