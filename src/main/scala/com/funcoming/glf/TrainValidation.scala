package com.funcoming.glf

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession

/**
  * Created by LiuFangGuo on 3/2/17.
  */
object TrainValidation {
  def main(args: Array[String]): Unit = {


    val sparkConf = new SparkConf().setMaster("local[*]").setAppName("交叉验证")
    val sparkContext = new SparkContext(sparkConf)
    val sparkSession = SparkSession.builder().appName("是不是session").getOrCreate()


    val data = sparkSession.read.format("libsvm").load("/Users/LiuFangGuo/Documents/SoftWare/spark-2.0.2-bin-hadoop2.6/data/mllib/sample_linear_regression_data.txt")
    //把一份数据随机分成training和test
    val Array(training, test) = data.randomSplit(Array(0.9, 0.1), seed = 12345)


    val lr = new LinearRegression()

    // We use a ParamGridBuilder to construct a grid of parameters to search over.
    // TrainValidationSplit will try all combinations of values and determine best model using
    // the evaluator.
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.1, 0.01))
      .addGrid(lr.fitIntercept)
      .addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0))
      .build()

    //TrainRatio把training数据再次分割。。。
    val trainValidationSplit = new TrainValidationSplit().setEstimator(lr).setEvaluator(new RegressionEvaluator()).setEstimatorParamMaps(paramGrid).setTrainRatio(0.8)

    val model = trainValidationSplit.fit(training)

    model.transform(test).select("features", "label", "prediction")
      .show()
    sparkSession.stop()

  }

}
