// See README.md for license details.

ThisBuild / scalaVersion := "2.13.14"
ThisBuild / version      := "0.1.0"
ThisBuild / organization := "be.kuleuven.esat.micas"

val chiselVersion   = "6.4.0"
val playJSONVersion = "3.0.4"

lazy val root = (project in file("."))
  .settings(
    name := "snax-streamer",
    libraryDependencies ++= Seq(
      "org.scala-lang"     % "scala-compiler" % "2.13.14",
      "org.chipsalliance" %% "chisel"         % chiselVersion,
      "edu.berkeley.cs"   %% "chiseltest"     % "6.0.0" % "test",
      "org.playframework" %% "play-json"      % playJSONVersion
    ),
    scalacOptions ++= Seq(
      "-language:reflectiveCalls",
      "-deprecation",
      "-feature",
      "-Xcheckinit",
      "-Ymacro-annotations",
      "-Wunused" // Enable unused import fixes

    ),
    addCompilerPlugin(
      "org.chipsalliance" % "chisel-plugin" % chiselVersion cross CrossVersion.full
    )
  )
