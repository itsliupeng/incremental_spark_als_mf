package com.example.als

import com.github.fommil.netlib.BLAS.{getInstance => blas}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkConf
import org.apache.spark.ml.recommendation.{ALSModel, MyALS, MyALSModel}
import org.apache.spark.sql.SparkSession

object ExampleLocal {
  import org.apache.spark.ml.evaluation.RegressionEvaluator

  val userCol = "userId"
  val itemCol = "itemId"
  val ratingCol = "rating"
  val predictionCol = "prediction"

  case class Rating(userId: Int, itemId: Int, rating: Float, time: Long)
  def parseRating(str: String): Rating = {
    val fields = str.split("::")
    assert(fields.size == 4)
    Rating(fields(0).toInt, fields(1).toInt, fields(2).toFloat, fields(3).toLong)
  }

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName(s"mag-${this.getClass.getName}")
      .setMaster("local")
      .set("spark.sql.parquet.compression.codec", "snappy")
    val spark = SparkSession.builder.config(conf).getOrCreate()



    implicit val sc = spark.sparkContext
    sc.setLogLevel("WARN")
    import spark.implicits._

    val work_dir = "/Users/liupeng/Dropbox/github/spark/data/mllib/als"

    sc.setCheckpointDir(work_dir + "/checkpoint")
    val ratings = spark.read.textFile(work_dir + "/sample_movielens_ratings.txt")
      .map(parseRating)
      .toDF()
    val Array(training, test) = ratings.randomSplit(Array(0.8, 0.2), 1023L)



    val numUserBlock = 10
    val numItemBlock = 10
    val rank = 10
    val maxIter = 5
    val checkPointInterval = 3

    val previousModel: MyALSModel = MyALSModel.load("/Users/liupeng/Dropbox/github/spark/data/mllib/als/model")
    // Build the recommendation model using ALS on the training data
    val als: MyALS = new MyALS(previousModel)
      .setRank(rank)
      .setMaxIter(maxIter)
      .setRegParam(0.01)
      .setUserCol(userCol)
      .setItemCol(itemCol)
      .setRatingCol(ratingCol)
      .setPredictionCol(predictionCol)
      .setCheckpointInterval(checkPointInterval)
      .setNumUserBlocks(numUserBlock)
      .setNumItemBlocks(numItemBlock)

    val model: ALSModel = als.fit(training)
    model.save((work_dir + "/new_model").overwritePath)

    // Evaluate the model by computing the RMSE on the test data
    // Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("rating")
      .setPredictionCol("prediction")


    val predictions = model.transform(test).na.drop("all", Seq(predictionCol))
    val rmse = evaluator.evaluate(predictions)
    println(s"Root-mean-square error = $rmse")


    val predictions1 = previousModel.transform(test).na.drop("all", Seq(predictionCol))
    val rmse1 = evaluator.evaluate(predictions1)
    println(s"original test = $rmse1")
    val predictions2 = previousModel.transform(training).na.drop("all", Seq(predictionCol))
    val rmse2 = evaluator.evaluate(predictions2)
    println(s"original train = $rmse2")


    val userRecs = model.asInstanceOf[MyALSModel].recommendForAllUsers(10)
    userRecs.show

    println("Done")
  }

  implicit class OverwritePath[T <: String](pathStr: String) {
    def overwritePath = {
      val path = new Path(pathStr)
      val fs = path.getFileSystem(new Configuration())
      if (fs.exists(path)) {
        fs.delete(path, true)
      }
      pathStr
    }
  }

}


