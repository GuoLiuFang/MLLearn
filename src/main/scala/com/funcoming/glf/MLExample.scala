package com.funcoming.glf

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.{Row, SparkSession}

/**
  * Created by LiuFangGuo on 2/28/17.
  */
object MLExample {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local[*]").setAppName("localtest")
    //    val conf = new SparkConf().setMaster("spark://killsong:7077").setAppName("standalonetest")
    //    val conf = new SparkConf().setMaster("spark://NY-HADOOP-12-151:7777").setAppName("线上test")
    val sc = new SparkContext(conf)


    val sparkSession = SparkSession.builder.appName("测试mllib").getOrCreate()
    //    Prepare training data from a list of (label, features) tuples.
    val traningDataSet = sparkSession.createDataFrame(Seq(
      (1.0, Vectors.dense(0.0, 1.1, 0.1)),
      (0.0, Vectors.dense(2.0, 1.0, -1.0)),
      (0.0, Vectors.dense(2.0, 1.3, 1.0)),
      (1.0, Vectors.dense(0.0, 1.2, -0.5))
    )).toDF("label", "features")

    val logisticRegression = new LogisticRegression()
    println("逻辑回归的参数是" + logisticRegression.explainParams())

    logisticRegression.setMaxIter(10).setRegParam(0.01)

    val logisticRegressionModel = logisticRegression.fit(traningDataSet)

    println("生成的这么模型的参数如下" + logisticRegressionModel.parent.extractParamMap())

    //We may alternatively specify parameters using a ParamMap
    val paramMap = ParamMap(logisticRegression.maxIter -> 20).put(logisticRegression.maxIter, 30).put(logisticRegression.regParam -> 0.1, logisticRegression.threshold -> 0.55)

    val paramMap2 = ParamMap(logisticRegression.probabilityCol -> "我设置的的概率标签")
    val paraMapCombine = paramMap ++ paramMap2

    val logisticRegressionModel2 = logisticRegression.fit(traningDataSet, paraMapCombine)

    println("模型2的参数是" + logisticRegressionModel2.parent.extractParamMap())

    //Prepare test data.
    val testDataSet = sparkSession.createDataFrame(Seq(
      (1.0, Vectors.dense(-1.0, 1.5, 1.3)),
      (0.0, Vectors.dense(3.0, 2.0, -0.1)),
      (1.0, Vectors.dense(0.0, 2.2, -1.5))
    )).toDF("label", "features")


    logisticRegressionModel2.transform(testDataSet).select("features", "label", "我设置的的概率标签", "prediction").collect()
      .foreach { case Row(feaglftest: Vector, testglftag: Double, prob: Vector, d: Double) => println(s"($feaglftest,$testglftag) -> 概率=$prob, 预测=$d") }

    sparkSession.stop()
  } //main


}
