package io

import java.io.File

import com.jmatio.io.MatFileReader
import com.jmatio.types.MLArray
import org.scalatest.FunSuite

/**
  * Created by Tom Lous on 13/08/2017.
  * Copyright © 2017 Datlinq B.V..
  */
class MatLabFileTest extends FunSuite {

  test("Read matlab file"){
    val inputFileName = "/dataset1_test.mat"
    val inputFile:File = new File(getClass.getResource(inputFileName).toURI)


    val matFileReader = new MatFileReader(inputFile)
    val matFileContent = matFileReader.getContent


    assert(matFileContent.containsKey("neg_examples_nobias"))
    assert(matFileContent.containsKey("pos_examples_nobias"))
    assert(matFileContent.containsKey("w_init"))
    assert(matFileContent.containsKey("w_gen_feas"))

    assert(matFileContent.get("neg_examples_nobias").getDimensions.toList === List(4,2))
    assert(matFileContent.get("pos_examples_nobias").getDimensions.toList === List(4,2))
    assert(matFileContent.get("w_init").getDimensions.toList === List(3,1))
    assert(matFileContent.get("w_gen_feas").getDimensions.toList === List(3,1))
  }

}

