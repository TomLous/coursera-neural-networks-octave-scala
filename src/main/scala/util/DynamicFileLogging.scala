package util

import java.io.File

import ch.qos.logback.classic.LoggerContext
import ch.qos.logback.classic.encoder.PatternLayoutEncoder
import ch.qos.logback.classic.spi.ILoggingEvent
import ch.qos.logback.core.FileAppender
import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory
/**
  * Created by Tom Lous on 27/08/2017.
  */
case class DynamicFileLogging(fileName: String, pattern:String) {

//   val BASIC_LOGGING = "%msg%n"
//  private val DEFAULT_PATTERN = "%date %level [%thread] %logger{10} [%file:%line] %msg%n"

//   lazy val logger: Logger =
//    Logger(LoggerFactory.getLogger(getClass.getName))


  val file: File = new File(fileName)
  val parent: File = file.getParentFile
  if (parent != null) parent.mkdirs

  val loggerContext = LoggerFactory.getILoggerFactory().asInstanceOf[LoggerContext]
  val loggerObj = loggerContext.getLogger(getClass.getName)

  // Setup pattern
  val patternLayoutEncoder = new PatternLayoutEncoder()
  patternLayoutEncoder.setPattern(pattern)
  patternLayoutEncoder.setContext(loggerContext)
  patternLayoutEncoder.start()

  // Setup appender
  val fileAppender = new FileAppender[ILoggingEvent]()
  fileAppender.setFile(fileName)
  fileAppender.setEncoder(patternLayoutEncoder)
  fileAppender.setContext(loggerContext)
  fileAppender.start()

  // Attach appender to logger
  loggerObj.addAppender(fileAppender)
  //logger.setLevel(Level.DEBUG)
//  logger.setAdditive(additive)

  fileAppender.getName


  lazy val logger: Logger = Logger(loggerObj)
//  val loggerContext: LoggerContext = LoggerFactory.getILoggerFactory.asInstanceOf[LoggerContext]
//
//  val fileAppender = new FileAppender[_]
//  fileAppender.setContext(loggerContext)
//  fileAppender.setName("timestamp")
//  // set the file name
//  fileAppender.setFile("log/" + System.currentTimeMillis + ".log")
//
//  val encoder:PatternLayoutEncoder = new PatternLayoutEncoder
//  encoder.setContext(loggerContext)
//  encoder.setPattern("%r %thread %level - %msg%n")
//  encoder.start()
//
//
//  LoggerFactory.
////  fileAppender.setEncoder(encoder)
//  fileAppender.start()
//
//  // attach the rolling file appender to the logger of your choice
//  val logbackLogger = loggerContext.getLogger("Main")
//  logbackLogger.addAppender(fileAppender)

}

object DynamicFileLogging{
  private val BASIC_LOGGING = "%msg%n"
  //  private val DEFAULT_PATTERN = "%date %level [%thread] %logger{10} [%file:%line] %msg%n"

  def apply(fileName: String):DynamicFileLogging = apply(fileName, BASIC_LOGGING)
}
