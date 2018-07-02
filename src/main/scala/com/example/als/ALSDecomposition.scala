package com.example.als

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.recommendation.{ALSModel, MyALS, MyALSModel}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{ArrayType, FloatType, StringType, StructField, StructType}

object ALSDecomposition {
  def main(args: Array[String]): Unit = {
    val executorNum = args(0)
    val rank = args(1).toInt
    val numUserBlock = args(2).toInt
    val numItemBlock = args(3).toInt
    val isFromScratch = if (args.length > 4) args(4).toBoolean else false

    val conf = new SparkConf().setAppName(s"als-${this.getClass.getName}")
      .set("spark.sql.parquet.compression.codec", "snappy")
      .set("spark.sql.shuffle.partitions", executorNum)

    val spark = SparkSession.builder.config(conf).getOrCreate()
    implicit val sc = spark.sparkContext
    import spark.implicits._

    val projectTime = System.currentTimeMillis() / 1000

    // hdfs path
    val data_path = "/users/liupeng/als"

    val outPathPrefix = s"$data_path/time${projectTime}_$today"
    val checkFile = s"$data_path/checkpoint/rank${rank}_time$projectTime"
    sc.setCheckpointDir(checkFile)

    //    attention: take it for granted that rating DataFrame is very clean, with no duplicate row of user_item
    val ratingPath = s"$data_path/rating/date=${yesterday}*"
    val ratingSource = spark.read.parquet(ratingPath).select($"user", $"item", $"rating".cast("float"), $"timestamp").filter($"timestamp" < getTimestamp(yesterday, daysAfter = 1)) // prevent wrong timestamp

    var userPath = s"$data_path/model/userFactors/date=${yesterday}*"
    var itemPath = s"$data_path/model/itemFacotrs/date=${yesterday}*"

    val trainIter = Seq((300, 3), (100, 3), (30, 3), (7, 2), (3, 1), (1, 1))
    val checkPointInterval = 10

    trainIter.zipWithIndex foreach { case ((days, times), index) =>
      clearAllCacche(sc)

      val thisRating = ratingRecentDays(ratingSource, yesterday, days)
      val user = thisRating.select($"user").distinct().sort($"user").as[User].rdd.zipWithIndex.map { case (user, index) => (user, index.toInt) }.toDF("user", "userId")
      user.cache()
      val item = thisRating.select($"item").distinct().sort($"item").as[Item].rdd.zipWithIndex.map { case (item, index) => (item, index.toInt) }.toDF("item", "itemId")
      item.cache()
      val thisTraining = thisRating.join(user, "user").join(item, "item").select("userId", "itemId", "rating", "timestamp")
      thisTraining.cache()

      // will cut dependency
      val (realUserFactorsAll, realItemFactorsAll) = if (isFromScratch) {
        (null, null)
      } else {
        (spark.read.parquet(userPath).toDF("user", "features"), spark.read.parquet(itemPath).toDF("item", "features"))
      }

      var model: ALSModel = MyALSModel.generateModel(rank, user, item)(realUserFactorsAll, realItemFactorsAll)

      var trainingRmse: Double = 0
      model = new MyALS(previousModel = model, monitorFile = None)
        .setRank(rank)
        .setMaxIter(times)
        .setRegParam(0.01)
        .setUserCol(userCol)
        .setItemCol(itemCol)
        .setRatingCol(ratingCol)
        .setPredictionCol(predictionCol)
        .setNumUserBlocks(numUserBlock)
        .setNumItemBlocks(numItemBlock)
        .setCheckpointInterval(checkPointInterval)
        .fit(thisTraining)

      trainingRmse = if (index == 0) {
        rmse(model, thisTraining)
      } else {
        0.0D
      }

      val realUserFactors = model.userFactors.select($"id".as("userId"), $"features".as("f1")).join(user, "userId").select("user", "f1")
      val realItemFactors = model.itemFactors.select($"id".as("itemId"), $"features".as("f1")).join(item, "itemId").select("item", "f1")
      import org.apache.spark.sql.functions.coalesce
      val realUserFactorsAllNew = realUserFactorsAll.join(realUserFactors, Seq("user"), "full").select($"user", coalesce($"f1", $"features")).toDF("user", "features")
      val realItemFactorsAllNew = realItemFactorsAll.join(realItemFactors, Seq("item"), "full").select($"item", coalesce($"f1", $"features")).toDF("item", "features")

      userPath = s"$data_path/model/userFactors/date=${today}_rmse_$trainingRmse"
      itemPath = s"$data_path/model/itemFactors/date=${today}_rmse_$trainingRmse"
      realUserFactorsAllNew.write.mode("overwrite").parquet(userPath)
      realItemFactorsAllNew.write.mode("overwrite").parquet(itemPath)
    }

    deleteCheckFile(checkFile)
    spark.stop()
  }

  def clearAllCacche(sc: SparkContext) = {
    sc.getPersistentRDDs.map { case (_, rdd) =>
      rdd.unpersist(true)
    }
  }
}


