package com.example

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALSModel
import org.apache.spark.sql.{DataFrame, Dataset}


package object als {
  type Item = String
  type User = String
  
  val userCol = "userId"
  val itemCol = "itemId"
  val ratingCol = "rating"
  val predictionCol = "prediction"

  val DATE_FORMAT = "yyyyMMdd"
  def dayBefore(n: Int): String = new org.joda.time.DateTime().minusDays(n).toString(DATE_FORMAT)
  val today = dayBefore(0)
  val yesterday = dayBefore(1)
  
  
  case class TimeRating(user: User, item: Item, rating: Float, timestamp: Long)
  
  // "20170101" => "2017-07-01"
  private def toDateFormat(s: String) = {
    require(s.length == 8)
    s"${s.substring(0, 4)}-${s.substring(4, 6)}-${s.substring(6)}"
  }

  // "20170101"
  def getTimestamp(s: String, daysAfter: Int = 0): Long = {
    val formatDate = toDateFormat(s)
    org.joda.time.DateTime.parse(formatDate).minusDays(-daysAfter).getMillis
  }
  
  def ratingRecentDays(data: DataFrame, latestDate: String, daysAfter: Int): DataFrame = {
    import data.sparkSession.implicits._
    val start  = getTimestamp(latestDate, 1) - 3600L * 24 * daysAfter * 1000
    data.filter($"timestamp" > start)
  }

  def rmse(model : ALSModel, input: Dataset[_]) = {
    val predictions = model.transform(input).na.drop("all", Seq(predictionCol))
    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol(ratingCol)
      .setPredictionCol(predictionCol)
    evaluator.evaluate(predictions)
  }


  def deleteCheckFile(checkFile: String): Unit = {
    Option(checkFile).map { file =>
      val path = new Path(file)
      val fs = path.getFileSystem(new Configuration())
      if (fs.exists(path)) {
        fs.delete(path, true)
      }
    }
  }
}
