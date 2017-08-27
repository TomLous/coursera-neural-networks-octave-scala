package util

import ch.qos.logback.classic.LoggerContext
import ch.qos.logback.classic.encoder.PatternLayoutEncoder
import ch.qos.logback.core.FileAppender
import org.slf4j.LoggerFactory
/**
  * Created by Tom Lous on 27/08/2017.
  */
class DynamicFileLogging {




  val loggerContext: LoggerContext = LoggerFactory.getILoggerFactory.asInstanceOf[LoggerContext]

  val fileAppender = new FileAppender[_]
  fileAppender.setContext(loggerContext)
  fileAppender.setName("timestamp")
  // set the file name
  fileAppender.setFile("log/" + System.currentTimeMillis + ".log")

  val encoder:PatternLayoutEncoder = new PatternLayoutEncoder
  encoder.setContext(loggerContext)
  encoder.setPattern("%r %thread %level - %msg%n")
  encoder.start()

//  fileAppender.setEncoder(encoder)
  fileAppender.start()

  // attach the rolling file appender to the logger of your choice
  val logbackLogger = loggerContext.getLogger("Main")
  logbackLogger.addAppender(fileAppender)
}
