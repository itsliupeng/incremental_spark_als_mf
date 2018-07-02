package org.apache.spark.ml.recommendation

import java.io.IOException
import java.{util => ju}

import com.github.fommil.netlib.BLAS.{getInstance => blas}
import org.apache.hadoop.fs.Path
import org.apache.spark.{Partitioner, SparkContext}
import org.apache.spark.annotation.Since
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.util.Instrumentation
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.{col, lit}
import org.apache.spark.sql.types.{ArrayType, FloatType, IntegerType, StringType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, KeyValueGroupedDataset}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.BoundedPriorityQueue
import org.apache.spark.util.collection.{OpenHashMap, OpenHashSet}
import org.apache.spark.util.random.XORShiftRandom
import org.slf4j.LoggerFactory

import scala.collection.mutable
import scala.reflect.ClassTag
import scala.util.Sorting
import scala.util.hashing.byteswap64

case class Record(name: String, iter: Int, time: Long, Score: Double)

object MyALS {
  val log = LoggerFactory.getLogger(this.getClass)

  def cleanDependency(rdd: RDD[_]) = {
    val previousCheckpointFile: Option[String] = rdd.getCheckpointFile
    val deps = rdd.dependencies
    rdd.checkpoint()
    rdd.count() // checkpoint item factors and cut lineage
    ALS.cleanShuffleDependencies(rdd.sparkContext, deps)
    deletePreviousCheckpointFile(previousCheckpointFile, rdd.sparkContext)
  }

  def deletePreviousCheckpointFile(previousCheckpointFile: Option[String], sc: SparkContext) = {
    previousCheckpointFile.foreach { file =>
      try {
        val checkpointFile = new Path(file)
        checkpointFile.getFileSystem(sc.hadoopConfiguration).delete(checkpointFile, true)
      } catch {
        case e: IOException =>
          log.error(s"Cannot delete checkpoint file $file:", e)
      }
    }
  }
}

class MyALS(previousModel: ALSModel = null, monitorFile: Option[String] = None) extends ALS {
  import org.apache.spark.ml.recommendation.ALS._

  override def fit(training: Dataset[_]): ALSModel = {
    transformSchema(training.schema)

    val r = if ($(ratingCol) != "") col($(ratingCol)).cast(FloatType) else lit(1.0f)
    val ratings = training
      .select(checkedCast(col($(userCol))), checkedCast(col($(itemCol))), r)
      .rdd
      .map { row =>
        Rating(row.getInt(0), row.getInt(1), row.getFloat(2))
      }

    val instr = Instrumentation.create(this, ratings)
    instr.logParams(rank, numUserBlocks, numItemBlocks, alpha, userCol,
      itemCol, ratingCol, predictionCol, maxIter, regParam, nonnegative, checkpointInterval,
      seed, intermediateStorageLevel, finalStorageLevel)

    val isContinuous =  if (previousModel == null) {
      false
    } else {
      true
    }
    val (userFactors, itemFactors) = myTrain(ratings, training, rank = $(rank),
      numUserBlocks = $(numUserBlocks), numItemBlocks = $(numItemBlocks),
      maxIter = $(maxIter), regParam = $(regParam),
      alpha = $(alpha), nonnegative = $(nonnegative),
      intermediateRDDStorageLevel = StorageLevel.fromString($(intermediateStorageLevel)),
      finalRDDStorageLevel = StorageLevel.fromString($(finalStorageLevel)),
      checkpointInterval = $(checkpointInterval), seed = $(seed), isContinuous = isContinuous)
    import training.sparkSession.implicits._
    val userDF = userFactors.toDF("id", "features")
    val itemDF = itemFactors.toDF("id", "features")
    val model: ALSModel = new ALSModel(uid, $(rank), userDF, itemDF)
    instr.logSuccess(model)
    copyValues(model)
  }

  type ALSPartitioner = org.apache.spark.HashPartitioner


  def getFactors(userF: DataFrame, itemF: DataFrame, userInBlocks: RDD[(Int, InBlock[Int])], itemInBlocks: RDD[(Int, InBlock[Int])]): (RDD[(Int, FactorBlock)], RDD[(Int, FactorBlock)]) = {
    import userF.sparkSession.implicits._
    val userRDD: RDD[(Int, Array[Float])] = userF.as[(Int, Array[Float])].rdd // user size is large
    val idBlockId = userInBlocks.flatMap { case (blockId, block) =>
      block.srcIds.map (i => (i, blockId))
    }

    val blockIdFactor = idBlockId.join(userRDD).map { case (id, (blockId, a)) =>
      (blockId, (id, a))
    }.groupByKey(new ALSPartitioner(userInBlocks.getNumPartitions))

    val userFactors: RDD[(Int, FactorBlock)] = userInBlocks.join(blockIdFactor).map { case (blockId, (block, userIdFactor)) =>
      val userIdFactorMap = userIdFactor.toMap
      val srcIds: Array[Int] = block.srcIds
      val factors: FactorBlock = srcIds.map(i => userIdFactorMap(i))
      (blockId, factors)
    }

    val item: Map[Int, Array[Float]] = itemF.as[(Int, Array[Float])].collect().toMap // item size is small
    val itemBr = itemF.sparkSession.sparkContext.broadcast(item)
    val itemFactors: RDD[(Int,FactorBlock)] = itemInBlocks.mapPartitions({ p =>
      p.map { case (blockId, block) =>
        val factorBlock: FactorBlock = block.srcIds.map(i => itemBr.value(i))
        (blockId, factorBlock)
      }
    }, preservesPartitioning = true)

    (userFactors, itemFactors)
  }

  def myTrain ( // scalastyle:ignore
                ratings: RDD[Rating[Int]],
                training: Dataset[_],
                rank: Int = 10,
                numUserBlocks: Int = 10,
                numItemBlocks: Int = 10,
                maxIter: Int = 10,
                regParam: Double = 0.1,
                alpha: Double = 1.0,
                nonnegative: Boolean = false,
                intermediateRDDStorageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK,
                finalRDDStorageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK,
                checkpointInterval: Int = 10,
                seed: Long = 0L,
                isContinuous: Boolean = false )(
                implicit ord: Ordering[Int]): (RDD[(Int, Array[Float])], RDD[(Int, Array[Float])]) = {
    require(intermediateRDDStorageLevel != StorageLevel.NONE,
      "ALS is not designed to run without persisting intermediate RDDs.")
    val sc = ratings.sparkContext
    val userPart = new ALSPartitioner(numUserBlocks)
    val itemPart = new ALSPartitioner(numItemBlocks)
    val userLocalIndexEncoder = new LocalIndexEncoder(userPart.numPartitions)
    val itemLocalIndexEncoder = new LocalIndexEncoder(itemPart.numPartitions)
    val solver = if (nonnegative) new NNLSSolver else new CholeskySolver
    val blockRatings: RDD[((Int, Int), RatingBlock[Int])] = partitionRatings(ratings, userPart, itemPart)
      .persist(intermediateRDDStorageLevel)
    val (userInBlocks: RDD[(Int, InBlock[Int])], userOutBlocks) =
      makeBlocks("user", blockRatings, userPart, itemPart, intermediateRDDStorageLevel)
    // materialize blockRatings and user blocks
//    userOutBlocks.count()
    val swappedBlockRatings = blockRatings.map {
      case ((userBlockId, itemBlockId), RatingBlock(userIds, itemIds, localRatings)) =>
        ((itemBlockId, userBlockId), RatingBlock(itemIds, userIds, localRatings))
    }
    val (itemInBlocks: RDD[(Int, InBlock[Int])], itemOutBlocks) =
      makeBlocks("item", swappedBlockRatings, itemPart, userPart, intermediateRDDStorageLevel)
    // materialize item blocks
//    itemOutBlocks.count()
    val seedGen = new XORShiftRandom(seed)
    var (userFactors, itemFactors) = if (isContinuous) {
      getFactors(previousModel.userFactors, previousModel.itemFactors, userInBlocks, itemInBlocks)
    } else {
      (initialize(userInBlocks, rank, seedGen.nextLong()), initialize(itemInBlocks, rank, seedGen.nextLong()))
    }
    var previousCheckpointFile: Option[String] = None
    val shouldCheckpoint: Int => Boolean = (iter) =>
      sc.checkpointDir.isDefined && checkpointInterval != -1 && (iter > 0 && iter % checkpointInterval == 0)
    val deletePreviousCheckpointFile: () => Unit = () =>
      previousCheckpointFile.foreach { file =>
        try {
          val checkpointFile = new Path(file)
          checkpointFile.getFileSystem(sc.hadoopConfiguration).delete(checkpointFile, true)
        } catch {
          case e: IOException =>
            logWarning(s"Cannot delete checkpoint file $file:", e)
        }
      }

    for (iter <- 0 until maxIter) {
      itemFactors = computeFactors(userFactors, userOutBlocks, itemInBlocks, rank, regParam,
        userLocalIndexEncoder, solver = solver)
      if (shouldCheckpoint(iter)) {
        // monitor loss
        monitorFile.map { file =>
          val value = calculateRmse(userFactors, itemFactors, training, rank, userInBlocks, itemInBlocks, finalRDDStorageLevel)
          import training.sparkSession.implicits._
          sc.makeRDD(Seq(Record("train", iter, System.currentTimeMillis() / 1000, value)))
            .coalesce(1)
            .toDF.write.mode("append").json(file)
        }

        val deps = itemFactors.dependencies
        itemFactors.checkpoint()
        itemFactors.count() // checkpoint item factors and cut lineage
        ALS.cleanShuffleDependencies(sc, deps)
        deletePreviousCheckpointFile()
        previousCheckpointFile = itemFactors.getCheckpointFile
      }
      userFactors = computeFactors(itemFactors, itemOutBlocks, userInBlocks, rank, regParam,
        itemLocalIndexEncoder, solver = solver)

    }
    val userIdAndFactors: RDD[(Int, Array[Float])] = userInBlocks
      .mapValues(_.srcIds)
      .join(userFactors)
      .mapPartitions({ items =>
        items.flatMap { case (_, (ids, factors)) =>
          ids.view.zip(factors)
        }
        // Preserve the partitioning because IDs are consistent with the partitioners in userInBlocks
        // and userFactors.
      }, preservesPartitioning = true)
      .setName("userFactors")
      .persist(finalRDDStorageLevel)
    val itemIdAndFactors: RDD[(Int, Array[Float])] = itemInBlocks
      .mapValues(_.srcIds)
      .join(itemFactors)
      .mapPartitions({ items =>
        items.flatMap { case (_, (ids, factors)) =>
          ids.view.zip(factors)
        }
      }, preservesPartitioning = true)
      .setName("itemFactors")
      .persist(finalRDDStorageLevel)
    if (finalRDDStorageLevel != StorageLevel.NONE) {
      userIdAndFactors.count()
      itemIdAndFactors.count()
//      MyALS.cleanDependency(userIdAndFactors)
//      MyALS.cleanDependency(itemIdAndFactors)
      itemFactors.unpersist()
      userInBlocks.unpersist()
      userOutBlocks.unpersist()
      itemInBlocks.unpersist()
      itemOutBlocks.unpersist()
      blockRatings.unpersist()
    }
    (userIdAndFactors, itemIdAndFactors)
  }

  private def calculateRmse(userFactors: RDD[(Int, FactorBlock)], itemFactors: RDD[(Int, FactorBlock)], training: Dataset[_], rank: Int, userInBlocks: RDD[(Int, InBlock[Int])], itemInBlocks: RDD[(Int, InBlock[Int])], finalRDDStorageLevel: StorageLevel): Double = {
    val userIdAndFactors: RDD[(Int, Array[Float])] = userInBlocks
      .mapValues(_.srcIds)
      .join(userFactors)
      .mapPartitions({ items =>
        items.flatMap { case (_, (ids, factors)) =>
          ids.view.zip(factors)
        }
        // Preserve the partitioning because IDs are consistent with the partitioners in userInBlocks
        // and userFactors.
      }, preservesPartitioning = true)
      .setName("userFactors")
      .persist(finalRDDStorageLevel)
    val itemIdAndFactors: RDD[(Int, Array[Float])] = itemInBlocks
      .mapValues(_.srcIds)
      .join(itemFactors)
      .mapPartitions({ items =>
        items.flatMap { case (_, (ids, factors)) =>
          ids.view.zip(factors)
        }
      }, preservesPartitioning = true)
      .setName("itemFactors")
      .persist(finalRDDStorageLevel)

    import training.sparkSession.implicits._
    val userDF = userIdAndFactors.toDF("id", "features")
    val itemDF = itemIdAndFactors.toDF("id", "features")
    val model = copyValues(new ALSModel(uid, rank, userDF, itemDF).setParent(this))
    val result = rmse(model, training)
    log.error(s"liupeng11 rmse: $result")
    result
  }

  def rmse(model : ALSModel, input: Dataset[_]): Double = {
    val predictions = model.transform(input).na.drop("all", Seq($(predictionCol)))
    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol($(ratingCol))
      .setPredictionCol($(predictionCol))
    evaluator.evaluate(predictions)
  }


  def partitionRatings[ID: ClassTag](
                                      ratings: RDD[Rating[ID]],
                                      srcPart: Partitioner,
                                      dstPart: Partitioner): RDD[((Int, Int), RatingBlock[ID])] = {

    /* The implementation produces the same result as the following but generates less objects.

    ratings.map { r =>
      ((srcPart.getPartition(r.user), dstPart.getPartition(r.item)), r)
    }.aggregateByKey(new RatingBlockBuilder)(
        seqOp = (b, r) => b.add(r),
        combOp = (b0, b1) => b0.merge(b1.build()))
      .mapValues(_.build())
    */

    val numPartitions = srcPart.numPartitions * dstPart.numPartitions
    ratings.mapPartitions { iter =>
      val builders = Array.fill(numPartitions)(new RatingBlockBuilder[ID])
      iter.flatMap { r =>
        val srcBlockId = srcPart.getPartition(r.user)
        val dstBlockId = dstPart.getPartition(r.item)
        val idx = srcBlockId + srcPart.numPartitions * dstBlockId
        val builder = builders(idx)
        builder.add(r)
        if (builder.size >= 2048) { // 2048 * (3 * 4) = 24k
          builders(idx) = new RatingBlockBuilder
          Iterator.single(((srcBlockId, dstBlockId), builder.build()))
        } else {
          Iterator.empty
        }
      } ++ {
        builders.view.zipWithIndex.filter(_._1.size > 0).map { case (block, idx) =>
          val srcBlockId = idx % srcPart.numPartitions
          val dstBlockId = idx / srcPart.numPartitions
          ((srcBlockId, dstBlockId), block.build())
        }
      }
    }.groupByKey().mapValues { blocks =>
      val builder = new RatingBlockBuilder[ID]
      blocks.foreach(builder.merge)
      builder.build()
    }.setName("ratingBlocks")
  }

  def makeBlocks[ID: ClassTag](
                                prefix: String,
                                ratingBlocks: RDD[((Int, Int), RatingBlock[ID])],
                                srcPart: Partitioner,
                                dstPart: Partitioner,
                                storageLevel: StorageLevel)(
                                implicit srcOrd: Ordering[ID]): (RDD[(Int, InBlock[ID])], RDD[(Int, OutBlock)]) = {
    val inBlocks = ratingBlocks.map {
      case ((srcBlockId, dstBlockId), RatingBlock(srcIds, dstIds, ratings)) =>
        // The implementation is a faster version of
        // val dstIdToLocalIndex = dstIds.toSet.toSeq.sorted.zipWithIndex.toMap
        val start = System.nanoTime()
        val dstIdSet = new OpenHashSet[ID](1 << 20)
        dstIds.foreach(dstIdSet.add)
        val sortedDstIds = new Array[ID](dstIdSet.size)
        var i = 0
        var pos = dstIdSet.nextPos(0)
        while (pos != -1) {
          sortedDstIds(i) = dstIdSet.getValue(pos)
          pos = dstIdSet.nextPos(pos + 1)
          i += 1
        }
        assert(i == dstIdSet.size)
        Sorting.quickSort(sortedDstIds)
        val dstIdToLocalIndex = new OpenHashMap[ID, Int](sortedDstIds.length)
        i = 0
        while (i < sortedDstIds.length) {
          dstIdToLocalIndex.update(sortedDstIds(i), i)
          i += 1
        }
        logDebug(
          "Converting to local indices took " + (System.nanoTime() - start) / 1e9 + " seconds.")
        val dstLocalIndices = dstIds.map(dstIdToLocalIndex.apply)
        (srcBlockId, (dstBlockId, srcIds, dstLocalIndices, ratings))
    }.groupByKey(new ALSPartitioner(srcPart.numPartitions))
      .mapValues { iter =>
        val builder =
          new UncompressedInBlockBuilder[ID](new LocalIndexEncoder(dstPart.numPartitions))
        iter.foreach { case (dstBlockId, srcIds, dstLocalIndices, ratings) =>
          builder.add(dstBlockId, srcIds, dstLocalIndices, ratings)
        }
        builder.build().compress()
      }.setName(prefix + "InBlocks")
      .persist(storageLevel)
    val outBlocks: RDD[(Int, Array[Array[Int]])] = inBlocks.mapValues { case InBlock(srcIds, dstPtrs, dstEncodedIndices, _) =>
      val encoder = new LocalIndexEncoder(dstPart.numPartitions)
      val activeIds = Array.fill(dstPart.numPartitions)(mutable.ArrayBuilder.make[Int])
      var i = 0
      val seen = new Array[Boolean](dstPart.numPartitions)
      while (i < srcIds.length) {
        var j = dstPtrs(i)
        ju.Arrays.fill(seen, false)
        while (j < dstPtrs(i + 1)) {
          val dstBlockId = encoder.blockId(dstEncodedIndices(j))
          if (!seen(dstBlockId)) {
            activeIds(dstBlockId) += i // add the local index in this out-block
            seen(dstBlockId) = true
          }
          j += 1
        }
        i += 1
      }
      activeIds.map { x =>
        x.result()
      }
    }.setName(prefix + "OutBlocks")
      .persist(storageLevel)
    (inBlocks, outBlocks)
  }

  def initialize[ID](
                      inBlocks: RDD[(Int, InBlock[ID])],
                      rank: Int,
                      seed: Long): RDD[(Int, FactorBlock)] = {
    // Choose a unit vector uniformly at random from the unit sphere, but from the
    // "first quadrant" where all elements are nonnegative. This can be done by choosing
    // elements distributed as Normal(0,1) and taking the absolute value, and then normalizing.
    // This appears to create factorizations that have a slightly better reconstruction
    // (<1%) compared picking elements uniformly at random in [0,1].
    inBlocks.map { case (srcBlockId, inBlock) =>
      val random = new XORShiftRandom(byteswap64(seed ^ srcBlockId))
      val factors = Array.fill(inBlock.srcIds.length) {
        val factor = Array.fill(rank)(random.nextGaussian().toFloat)
        val nrm = blas.snrm2(rank, factor, 1)
        blas.sscal(rank, 1.0f / nrm, factor, 1)
        factor
      }
      (srcBlockId, factors)
    }
  }

  type FactorBlock = Array[Array[Float]]
  type OutBlock = Array[Array[Int]]
  def computeFactors[ID](
                          srcFactorBlocks: RDD[(Int, FactorBlock)],
                          srcOutBlocks: RDD[(Int, OutBlock)],
                          dstInBlocks: RDD[(Int, InBlock[ID])],
                          rank: Int,
                          regParam: Double,
                          srcEncoder: LocalIndexEncoder,
                          implicitPrefs: Boolean = false,
                          alpha: Double = 1.0,
                          solver: LeastSquaresNESolver): RDD[(Int, FactorBlock)] = {
    val numSrcBlocks = srcFactorBlocks.partitions.length
    val srcOut = srcOutBlocks.join(srcFactorBlocks).flatMap {
      case (srcBlockId, (srcOutBlock, srcFactors)) =>
        srcOutBlock.view.zipWithIndex.map { case (activeIndices, dstBlockId) =>
          (dstBlockId, (srcBlockId, activeIndices.map(idx => srcFactors(idx))))
        }
    }
    val merged = srcOut.groupByKey(new ALSPartitioner(dstInBlocks.partitions.length))
    dstInBlocks.join(merged).mapValues {
      case (InBlock(dstIds, srcPtrs, srcEncodedIndices, ratings), srcFactors) =>
        val sortedSrcFactors = new Array[FactorBlock](numSrcBlocks)
        srcFactors.foreach { case (srcBlockId, factors) =>
          sortedSrcFactors(srcBlockId) = factors
        }
        val dstFactors = new Array[Array[Float]](dstIds.length)
        var j = 0
        val ls = new NormalEquation(rank)
        while (j < dstIds.length) {
          ls.reset()
          var i = srcPtrs(j)
          var numExplicits = 0
          while (i < srcPtrs(j + 1)) {
            val encoded = srcEncodedIndices(i)
            val blockId = srcEncoder.blockId(encoded)
            val localIndex = srcEncoder.localIndex(encoded)
            val srcFactor = sortedSrcFactors(blockId)(localIndex)
            val rating = ratings(i)
            ls.add(srcFactor, rating)
            numExplicits += 1
            i += 1
          }
          // Weight lambda by the number of explicit ratings based on the ALS-WR paper.
          dstFactors(j) = solver.solve(ls, numExplicits * regParam)
          j += 1
        }
        dstFactors
    }
  }

  def computeYtY(factorBlocks: RDD[(Int, FactorBlock)], rank: Int): NormalEquation = {
    factorBlocks.values.aggregate(new NormalEquation(rank))(
      seqOp = (ne, factors) => {
        factors.foreach(ne.add(_, 0.0))
        ne
      },
      combOp = (ne1, ne2) => ne1.merge(ne2))
  }
}

object MyALSModel {
  type Item = String
  type User = String

  def load(modelPath: String) = {
    val model = ALSModel.load(modelPath)
    new MyALSModel("1", model.rank, model.userFactors, model.itemFactors)
  }

  def generateModel(rank: Int, validUser: DataFrame, validItem: DataFrame)(previousUserFactors: DataFrame, previousItemFactors: DataFrame): ALSModel = {
    import previousUserFactors.sparkSession.implicits._
    val userFactorsRDD = if (previousUserFactors == null) {
      validUser.as[(User, Int)].map { case (_, userId) =>
        (userId, generateFactor(rank))
      }.rdd
    } else {
      val userFactorValid: Dataset[(Int, Array[Float])] = validUser.join(previousUserFactors, "user").select("userId", "features").as[(Int, Array[Float])]
      val userFactorRandom = validUser.join(previousUserFactors, Seq("user"), "left_anti").as[(User, Int)].map { case (_, userId) =>
        (userId, generateFactor(rank))
      }
      userFactorValid.rdd ++ userFactorRandom.rdd
    }
//    MyALS.cleanDependency(userFactorsRDD)
    val userFactors = userFactorsRDD.toDF("id", "features")

    val itemFactorsRDD = if (previousItemFactors == null) {
      validItem.as[(Item, Int)].map { case (_, itemId) =>
        (itemId, generateFactor(rank))
      }.rdd
    } else {
      val itemFactorValid = validItem.join(previousItemFactors, "item").select("itemId", "features").as[(Int, Array[Float])]
      val itemFactorRandom = validItem.join(previousItemFactors, Seq("item"), "left_anti").as[(Item, Int)].map { case (_, itemId) =>
        (itemId, generateFactor(rank))
      }

      itemFactorValid.rdd ++ itemFactorRandom.rdd
    }

//    MyALS.cleanDependency(itemFactorsRDD)
    val itemFactors: DataFrame = itemFactorsRDD.toDF("id", "features")

//    userFactors.count()
//    itemFactors.count()

    new ALSModel("1", rank, userFactors, itemFactors)
  }

  def generateModelFromAll(rank: Int, realUserFactors: DataFrame, realItemFactors: DataFrame, validUser: DataFrame, validItem: DataFrame): ALSModel = {
    val userFactors = validUser.join(realUserFactors, "user").select("userId", "features").toDF("id", "features")
    val itemFactors = validItem.join(realItemFactors, "item").select("itemId", "features").toDF("id", "features")
    new ALSModel("1", rank, userFactors, itemFactors)
  }

  val random = new XORShiftRandom(byteswap64(System.currentTimeMillis()))
  def generateFactor(rank: Int): Array[Float] = {
    val factor = Array.fill(rank)(random.nextGaussian().toFloat)
    val nrm = blas.snrm2(rank, factor, 1)
    blas.sscal(rank, 1.0f / nrm, factor, 1)
    factor
  }
}


class MyALSModel (
                   @Since("1.4.0") override val uid: String,
                   @Since("1.4.0") override val rank: Int,
                   @transient override val userFactors: DataFrame,
                   @transient override val itemFactors: DataFrame
                 ) extends ALSModel(uid, rank, userFactors, itemFactors) {

  def recommendForAllUsersBetter(numItems: Int, srcBlockSize: Int =  4096, dstBlockSize: Int = 4096): RDD[(Int, Seq[(Int, Float)])] = {
    recommendForAllBetter(userFactors, itemFactors, $(userCol), $(itemCol), numItems, srcBlockSize, dstBlockSize)
  }

  def recommendForAllUsersBetterBr(numItems: Int, srcBlockSize: Int =  4096, dstBlockSize: Int = 4096): Dataset[(Int, Seq[(Int, Float)])] = {
    recommendForAllBetterBr(userFactors, itemFactors, $(userCol), $(itemCol), numItems, srcBlockSize, dstBlockSize)
  }

  def recommendForAllUsersBetterString(numItems: Int): DataFrame = {
    recommendForAllBetterString(userFactors, itemFactors, $(userCol), $(itemCol), numItems)
  }

  def recommendForAllUsersString(numItems: Int): DataFrame = {
    recommendForAllWithString(userFactors, itemFactors, $(userCol), $(itemCol), numItems)
  }

  def recommendForAllUsers(numItems: Int): DataFrame = {
    recommendForAll(userFactors, itemFactors, $(userCol), $(itemCol), numItems)
  }

  def recommendForAllItems(numUsers: Int): DataFrame = {
    recommendForAll(itemFactors, userFactors, $(itemCol), $(userCol), numUsers)
  }


  def recommendForAll(
                       srcFactors: DataFrame,
                       dstFactors: DataFrame,
                       srcOutputColumn: String,
                       dstOutputColumn: String,
                       num: Int): DataFrame = {
    import srcFactors.sparkSession.implicits._

    val srcFactorsBlocked: Dataset[Seq[(Int, Array[Float])]] = blockify(srcFactors.as[(Int, Array[Float])])
    val dstFactorsBlocked: Dataset[Seq[(Int, Array[Float])]] = blockify(dstFactors.as[(Int, Array[Float])])
    val ratings: Dataset[(Int, Int, Float)] = srcFactorsBlocked.crossJoin(dstFactorsBlocked)
      .as[(Seq[(Int, Array[Float])], Seq[(Int, Array[Float])])]
      .flatMap { case (srcIter, dstIter) =>
        val m = srcIter.size
        val n = math.min(dstIter.size, num)
        val output = new Array[(Int, Int, Float)](m * n)
        var i = 0
        val pq = new BoundedPriorityQueue[(Int, Float)](num)(Ordering.by(_._2))
        srcIter.foreach { case (srcId, srcFactor) =>
          dstIter.foreach { case (dstId, dstFactor) =>
            // We use F2jBLAS which is faster than a call to native BLAS for vector dot product
//            val score = blas.sdot(rank, srcFactor, 1, dstFactor, 1)
//            val score = org.netlib.blas.Sdot.sdot(rank, srcFactor, 0, 1, dstFactor, 0, 1)
            val score = vectorMultiply(rank, srcFactor, dstFactor)
            pq += dstId -> score
          }
          pq.foreach { case (dstId, score) =>
            output(i) = (srcId, dstId, score)
            i += 1
          }
          pq.clear()
        }
        output.toSeq
      }
    // We'll force the IDs to be Int. Unfortunately this converts IDs to Int in the output.
    val topKAggregator = new TopByKeyAggregator[Int, Int, Float](num, Ordering.by(_._2))
    val g: KeyValueGroupedDataset[Int, (Int, Int, Float)] = ratings.as[(Int, Int, Float)].groupByKey(_._1)
    val recs = g.agg(topKAggregator.toColumn)
      .toDF("id", "recommendations")

    val arrayType = ArrayType(
      new StructType()
        .add(dstOutputColumn, IntegerType)
        .add("rating", FloatType)
    )
    recs.select($"id".as(srcOutputColumn), $"recommendations".cast(arrayType))
  }



  def recommendForAllBetter(
                       srcFactors: DataFrame,
                       dstFactors: DataFrame,
                       srcOutputColumn: String,
                       dstOutputColumn: String,
                       num: Int,
                       srcBlockSize: Int =  4096,
                       dstBlockSize: Int = 4096): RDD[(Int, Seq[(Int, Float)])] = {
    import srcFactors.sparkSession.implicits._

    val srcFactorsBlocked: Dataset[Seq[(Int, Array[Float])]] = blockify(srcFactors.as[(Int, Array[Float])], srcBlockSize)
    val dstFactorsBlocked: Dataset[Seq[(Int, Array[Float])]] = blockify(dstFactors.as[(Int, Array[Float])], dstBlockSize)
    val ratings: Dataset[(Int, Int, Float)] = srcFactorsBlocked.crossJoin(dstFactorsBlocked)
      .as[(Seq[(Int, Array[Float])], Seq[(Int, Array[Float])])]
      .flatMap { case (srcIter, dstIter) =>
        srcIter.flatMap { case (srcId, srcFactor) =>
          dstIter.map { case (dstId, dstFactor) =>
            val score = vectorMultiply(rank, srcFactor, dstFactor)
            (srcId, dstId, score)
          }.filter(_._3 > 0.5) // threshold is 0.5
        }
      }

    val result: RDD[(Int, Seq[(Int, Float)])] = ratings.rdd.groupBy(_._1).map { case (key, s) =>
      val sortedSeq = s.map(i => (i._2, i._3)).toSeq.sortWith(_._2 > _._2).slice(0, num)
      (key, sortedSeq)
    }
    result
  }

  def recommendForAllBetterBr(
                             srcFactors: DataFrame,
                             dstFactors: DataFrame,
                             srcOutputColumn: String,
                             dstOutputColumn: String,
                             num: Int,
                             srcBlockSize: Int = 4096,
                             dstBlockSize: Int = 4096): Dataset[(Int, Seq[(Int, Float)])] = {
    import srcFactors.sparkSession.implicits._

    val sc = srcFactors.sparkSession.sparkContext

    val dstFactorBr: Broadcast[Array[(Int, Array[Float])]] = sc.broadcast(dstFactors.as[(Int, Array[Float])].rdd.collect())

    val result: Dataset[(Int, Seq[(Int, Float)])] = srcFactors.as[(Int, Array[Float])].map { case (userId, userSeq) =>
      val dstFactor = dstFactorBr.value
      val value: Array[(Int, Float)] = dstFactor.map { case (itemId, itemSeq) =>
        val rank = userSeq.length
        (itemId, vectorMultiply(rank, userSeq, itemSeq))
      }.filter(_._2 > 0.5)
        .sortWith(_._2 > _._2)
        .slice(0, num)
      (userId, value)
    }
    result
  }


  def recommendForAllBetterString(
                             srcFactors: DataFrame,
                             dstFactors: DataFrame,
                             srcOutputColumn: String,
                             dstOutputColumn: String,
                             num: Int): DataFrame = {
    import srcFactors.sparkSession.implicits._

    val srcFactorsBlocked: Dataset[Seq[(String, Array[Float])]] = blockifyString(srcFactors.as[(String, Array[Float])])
    val dstFactorsBlocked = blockifyString(dstFactors.as[(String, Array[Float])])
    val ratings: Dataset[(String, String, Float)] = srcFactorsBlocked.crossJoin(dstFactorsBlocked)
      .as[(Seq[(String, Array[Float])], Seq[(String, Array[Float])])]
      .flatMap { case (srcIter, dstIter) =>
        srcIter.flatMap { case (srcId, srcFactor) =>
          dstIter.map { case (dstId, dstFactor) =>
            val score = vectorMultiply(rank, srcFactor, dstFactor)
            (srcId, dstId, score)
          }.filter(_._3 > 1.0)
        }
      }

    val result: RDD[(String, Seq[(String, Float)])] = ratings.rdd.groupBy(_._1).map { case (key, s) =>
      val sortedSeq = s.map(i => (i._2, i._3)).toSeq.sortWith(_._2 > _._2)
      (key, sortedSeq)
    }
    result.toDF(srcOutputColumn, dstOutputColumn)
  }



  def recommendForAllWithString(
                       srcFactors: DataFrame,
                       dstFactors: DataFrame,
                       srcOutputColumn: String,
                       dstOutputColumn: String,
                       num: Int): DataFrame = {
    import srcFactors.sparkSession.implicits._

    val srcFactorsBlocked: Dataset[Seq[(String, Array[Float])]] = blockifyString(srcFactors.as[(String, Array[Float])])
    val dstFactorsBlocked: Dataset[Seq[(String, Array[Float])]] = blockifyString(dstFactors.as[(String, Array[Float])])
    val ratings: Dataset[(String, String, Float)] = srcFactorsBlocked.crossJoin(dstFactorsBlocked)
      .as[(Seq[(String, Array[Float])], Seq[(String, Array[Float])])]
      .flatMap { case (srcIter, dstIter) =>
        val m = srcIter.size
        val n = math.min(dstIter.size, num)
        val output = new Array[(String, String, Float)](m * n)
        var i = 0
        val pq = new BoundedPriorityQueue[(String, Float)](num)(Ordering.by(_._2))
        srcIter.foreach { case (srcId, srcFactor) =>
          dstIter.foreach { case (dstId, dstFactor) =>
            // We use F2jBLAS which is faster than a call to native BLAS for vector dot product
//            val score = org.netlib.blas.Sdot.sdot(rank, srcFactor, 0, 1, dstFactor, 0, 1)
            val score = vectorMultiply(rank, srcFactor, dstFactor)
            pq += dstId -> score
          }
          pq.foreach { case (dstId, score) =>
            output(i) = (srcId, dstId, score)
            i += 1
          }
          pq.clear()
        }
        output.toSeq
      }
    // We'll force the IDs to be Int. Unfortunately this converts IDs to Int in the output.
    val topKAggregator = new TopByKeyAggregator[String, String, Float](num, Ordering.by(_._2))
    val g: KeyValueGroupedDataset[String, (String, String, Float)] = ratings.as[(String, String, Float)].groupByKey(_._1)
    val recs = g.agg(topKAggregator.toColumn)
      .toDF("id", "recommendations")

    val arrayType = ArrayType(
      new StructType()
        .add(dstOutputColumn, StringType)
        .add("rating", FloatType)
    )
    recs.select($"id".as(srcOutputColumn), $"recommendations".cast(arrayType))
  }

  @inline
  def vectorMultiply(rank: Int, u: Array[Float], v: Array[Float]) = {
//    0.until(rank).map { i =>
//      u(i) * v(i)
//    }.reduce(_ + _)

    var result = 0.0F
    0.until(rank).foreach { i =>
      result += u(i) * v(i)
    }
    result
  }



  def blockify(
                factors: Dataset[(Int, Array[Float])],
                blockSize: Int = 4096): Dataset[Seq[(Int, Array[Float])]] = {
    import factors.sparkSession.implicits._
    factors.mapPartitions(_.grouped(blockSize))
  }

  def blockifyString(
                factors: Dataset[(String, Array[Float])],
                blockSize: Int = 4096): Dataset[Seq[(String, Array[Float])]] = {
    import factors.sparkSession.implicits._
    factors.mapPartitions(_.grouped(blockSize))
  }
}

import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.expressions.Aggregator
import org.apache.spark.sql.{Encoder, Encoders}
import org.apache.spark.util.BoundedPriorityQueue

import scala.language.implicitConversions
import scala.reflect.runtime.universe.TypeTag

class TopByKeyAggregator[K1: TypeTag, K2: TypeTag, V: TypeTag]
(num: Int, ord: Ordering[(K2, V)])
  extends Aggregator[(K1, K2, V), BoundedPriorityQueue[(K2, V)], Array[(K2, V)]] {

  override def zero: BoundedPriorityQueue[(K2, V)] = new BoundedPriorityQueue[(K2, V)](num)(ord)

  override def reduce(
                       q: BoundedPriorityQueue[(K2, V)],
                       a: (K1, K2, V)): BoundedPriorityQueue[(K2, V)] = {
    q += {(a._2, a._3)}
  }

  override def merge(
                      q1: BoundedPriorityQueue[(K2, V)],
                      q2: BoundedPriorityQueue[(K2, V)]): BoundedPriorityQueue[(K2, V)] = {
    q1 ++= q2
  }

  override def finish(r: BoundedPriorityQueue[(K2, V)]): Array[(K2, V)] = {
    r.toArray.sorted(ord.reverse)
  }

  override def bufferEncoder: Encoder[BoundedPriorityQueue[(K2, V)]] = {
    Encoders.kryo[BoundedPriorityQueue[(K2, V)]]
  }

  override def outputEncoder: Encoder[Array[(K2, V)]] = ExpressionEncoder[Array[(K2, V)]]()
}

