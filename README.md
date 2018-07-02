## 在 Spark MLlib ALS 基础上实现的增量协同过滤矩阵分解

### 满足的场景

每天新增 rating 数据，对原有的 user embedding 和 item embedding 进行更新


### 性能

已在实际场景中使用，3kw+ user， 4w+ item，embedding， 维数 600。
executorNum 为 600，numUserBlock, numItemBlock 为 600。 矩阵分解 3 次约需 7 个小时
具体参数如下，

```
    --conf "spark.executor.extraJavaOptions=-XX:+PrintFlagsFinal -XX:+UseParallelOldGC -Xmn2200m -XX:SurvivorRatio=20 -XX:TargetSurvivorRatio=100 -XX:MaxTenuringThreshold=5 -XX:PermSize=96m -XX:MaxPermSize=96m -XX:ReservedCodeCacheSize=128m -XX:-UseBiasedLocking -XX:+ExplicitGCInvokesConcurrent -XX:+PrintTenuringDistribution -XX:PrintFLSStatistics=2 -XX:+PrintGCDetails -XX:+PrintSafepointStatistics -XX:+PrintGCDateStamps -XX:+PrintGCTimeStamps -XX:+PrintGCApplicationStoppedTime -XX:+PrintGCApplicationConcurrentTime -XX:+PrintPromotionFailure -XX:+HeapDumpOnOutOfMemoryError -XX:+UnlockDiagnosticVMOptions" \
    --conf "spark.serializer=org.apache.spark.serializer.KryoSerializer" \
    --conf "spark.kryoserializer.buffer=512m" \
    --conf "spark.blacklist.enabled=true" \
    --conf "spark.shuffle.compress=true" \
    --conf "spark.shuffle.spill.compress=true" \
    --conf "spark.shuffle.io.preferDirectBufs=true" \
    --conf "spark.shuffle.service.enabled=true" \
    --conf "spark.akka.frameSize=1024" \
    --conf "spark.memory.useLegacyMode=true" \
    --conf "spark.shuffle.memoryFraction=0.8" \
    --conf "spark.storage.memoryFraction=0.2" \
    --conf "spark.yarn.executor.memoryOverhead=2048" \
    --conf "spark.shuffle.spill.initialMemoryThreshold=5242880000" \
    --conf "spark.shuffle.memory.estimate.debug.enable=false" \
    --conf "spark.shuffle.spill.checkJvmHeap.enable=true" \
    --conf "spark.shuffle.spill.checkJvmHeap.oldSpacePercent=90" \
    --conf "spark.shuffle.spill.checkJvmHeap.logPercent=88" \
    --conf "spark.rdd.compress=true" \
    --conf "spark.broadcast.compress=true" \
    --conf "spark.driver.maxResultSize=4g" \
    --conf "spark.eventLog.enabled=true" \
    --master yarn-cluster \
    --driver-memory 6000m \
    --executor-memory 6000m \
```


