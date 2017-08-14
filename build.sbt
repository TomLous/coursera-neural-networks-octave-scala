name := "coursera-neural-networks-octave-scala"

version := "1.0"

scalaVersion := "2.12.1"

libraryDependencies ++= Seq(
  "com.diffplug.matsim"          % "matfilerw"                % "3.0.1",
  "com.typesafe"                 % "config"                   % "1.3.1",
  "com.typesafe.scala-logging"  %% "scala-logging"            % "3.7.2",
  "org.slf4j"                    % "slf4j-simple"             % "1.7.25",
  "org.scalanlp"                %% "breeze"                   % "0.13.2",
  "org.scalanlp"                %% "breeze-viz"               % "0.13.2",
  "org.scalatest"               %% "scalatest"                % "3.0.0"    % "test"
)


        