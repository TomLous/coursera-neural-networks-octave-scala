package io

import java.io.File
import java.net.URI


import breeze.linalg.DenseMatrix

import scala.collection.JavaConverters._
import com.jmatio.io.MatFileReader
import com.jmatio.types.MLArray

import scala.util.Try

/**
  * Created by Tom Lous on 14/08/2017.
  */


case class MatLabFile(filePath: URI) {
  type MatLabFileContents = Map[String, MLArray]

  lazy val contents: Either[Throwable, MatLabFileContents] = readFile

  private def readFile = {
    Try {
      val inputFile: File = new File(filePath)
      val matFileReader = new MatFileReader(inputFile)
      matFileReader.getContent.asScala.toMap
    }.toEither
  }

  def mlArray(name: String): Either[Throwable, MLArray] = {
    contents
      .right
      .flatMap(
        _.get(name) match {
          case Some(mlArray) => Right(mlArray)
          case None => Left(new Exception(s"MLArray `$name` not found"))
        })
  }

  def mlArrayOption(name: String): Option[MLArray] = mlArray(name) match {
    case Right(mlArray) => Some(mlArray)
    case Left(_) => None
  }


  def denseMatrixOption(name: String): Option[DenseMatrix[Double]] = {
    import util.MatLabConversions._

    for {
      mlArray <- mlArrayOption(name)
      denseMatrix <- mlArray
    } yield denseMatrix
  }
}


