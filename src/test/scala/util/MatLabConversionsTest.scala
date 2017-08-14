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


  test("mlArrayToDenseMatrix") { mlFile => {
    val mlArray = mlFile.mlArrayOption("neg_examples_nobias").get
    val denseMatrix: DenseMatrix[Double] = MatLabConversions.mlArrayToDenseMatrix(mlArray).get

    assert(mlArray.getM === denseMatrix.rows)
    assert(mlArray.getN === denseMatrix.cols)


  }
  }


  test("mlArrayToDenseMatrix implicit") { mlFile => {
    import MatLabConversions._

    val dd:Option[DenseMatrix[Double]] = for{
      mlArray <-  mlFile.mlArrayOption("neg_examples_nobias")
      denseMatrix <- mlArray
    } yield denseMatrix


    assert(dd.get.rows === 4)
    assert(dd.get.cols === 2)
  }
  }


  test("mlArrayToDenseMatrix from MLFile") { mlFile => {
    assert(mlFile.denseMatrixOption("neg_examples_nobias").get.rows === 4)
    assert(mlFile.denseMatrixOption("neg_examples_nobias").get.cols === 2)
  }
  }

}
