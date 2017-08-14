package util

import java.lang

import breeze.linalg.DenseMatrix
import io.MatLabFile
import org.scalatest.{BeforeAndAfterEach, FunSuite, fixture}
import collection.JavaConverters._

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

    //    println(double)


  }
  }

  //  test("raw converion") { mlFile =>{
  //
  //    val a: java.lang.Double = 3.4
  //    val b: java.lang.Double = 3.4
  //
  //    val l:IndexedSeq[IndexedSeq[java.lang.Double]] = IndexedSeq(IndexedSeq(a,a,a), IndexedSeq(b,b,b))
  //    println(DenseMatrix(l: _*))
  //
  //
  //  }}

  test("testMlArrayToDenseMatrix2") { mlFile => {
    val mlArray = mlFile.mlArrayOption("neg_examples_nobias").get
    val denseMatrix: DenseMatrix[Double] = MatLabConversions.mlArrayToDenseMatrix(mlArray).get


    assert(mlArray.getM === denseMatrix.rows)
    assert(mlArray.getN === denseMatrix.cols)


  }
  }

}
