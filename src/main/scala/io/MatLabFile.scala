package io

import java.io.File
import java.net.URI

import breeze.linalg.{DenseMatrix, DenseVector}
import com.jmatio.io.MatFileReader
import com.jmatio.types.{MLArray, MLStructure}

import scala.collection.JavaConverters._
import scala.util.Try

/**
  * Created by Tom Lous on 14/08/2017.
  */


case class MatLabFile(filePath: URI) {
  type MatLabFileContents = Map[String, MLArray]

  lazy val contents: Either[Throwable, MatLabFileContents] = readFile


  private def recursiveReadStruct(name: String, structure: MLStructure, prefix:List[String]=Nil):Map[String, MLArray] = {
    structure.getAllFields.asScala.toList.flatMap{
      case mlarr:MLStructure => recursiveReadStruct(mlarr.name, mlarr, name :: prefix)
      case mlarr => {
        val key:String = (mlarr.name :: name :: prefix).reverse.mkString(".")
        List(key -> mlarr)
      }
    }.toMap
  }

  private def readFile = {
    Try {
      val inputFile: File = new File(filePath)
      val matFileReader = new MatFileReader(inputFile)
      matFileReader.getContent.asScala.flatMap {
        case (name, structure: MLStructure) => { //@todo perhaps add recursion
          recursiveReadStruct(name, structure)
        }
        case (name, data) => Some(name -> data)
      }.toMap

    }.toEither
  }

  def mlArray(name: String): Either[Throwable, MLArray] = {
    contents
      .right
      .flatMap(
        _.get(name) match {
          case Some(mlArray) if !mlArray.isEmpty => Right(mlArray)
          case Some(_) => Left(new Exception(s"MLArray `$name` empty"))
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
      denseMatrix <- mlArray: Option[DenseMatrix[Double]]
    } yield denseMatrix
  }

  def denseVectorOption(name: String): Option[DenseVector[Double]] = {
    denseMatrixOption(name).map(_.toDenseVector)
  }

  def stringListOption(name: String): Option[List[String]] = {
    import util.MatLabConversions._

    for{
      mlArray <- mlArrayOption(name)
      list <- mlArray: Option[List[String]]
    } yield list

  }

  def listNames:Option[List[String]] = contents.map(_.keys.toList).toOption

}


