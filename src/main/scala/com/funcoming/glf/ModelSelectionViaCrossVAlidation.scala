package com.funcoming.glf

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.linalg.Vector

/**
  * Created by LiuFangGuo on 3/1/17.
  */
object ModelSelectionViaCrossVAlidation {
  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setMaster("local[*]").setAppName("交叉验证")
    val sparkContext = new SparkContext(sparkConf)
    val sparkSession = SparkSession.builder().appName("是不是session").getOrCreate()

    val trainDataSet = sparkSession.createDataFrame(Seq(
      (0L, "a b c d e spark", 1.0),
      (1L, "b d", 0.0),
      (2L, "spark f g h", 1.0),
      (3L, "hadoop mapreduce", 0.0),
      (4L, "b spark who", 1.0),
      (5L, "g d a y", 0.0),
      (6L, "spark fly", 1.0),
      (7L, "was mapreduce", 0.0),
      (8L, "e spark program", 1.0),
      (9L, "a e c l", 0.0),
      (10L, "spark compile", 1.0),
      (11L, "hadoop software", 0.0)
    )).toDF("id", "text", "label")

    val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")

    val hashingTF = new HashingTF().setInputCol(tokenizer.getOutputCol).setOutputCol("features")

    val logisticRegression = new LogisticRegression().setMaxIter(10)

    val pipeline = new Pipeline().setStages(Array(tokenizer, hashingTF, logisticRegression))

    val paramGrid = new ParamGridBuilder().addGrid(hashingTF.numFeatures, Array(10, 100, 1000)).addGrid(logisticRegression.regParam, Array(0.1, 0.01)).build()

    // We now treat the Pipeline as an Estimator, wrapping it in a CrossValidator instance..This will allow us to jointly choose parameters for all Pipeline stages...
    // A crossvalidator requires an Estimator , a set of Estimator paramaps and an Evaluator Note that the evaluator here is a binaryClassificaitonEvaluator
    val crossValidator = new CrossValidator().setEstimator(pipeline).setEvaluator(new BinaryClassificationEvaluator()).setEstimatorParamMaps(paramGrid).setNumFolds(2)
    val crossValidatorModel = crossValidator.fit(trainDataSet)

    val testDataSet = sparkSession.createDataFrame(Seq(
      (4L, "spark i j k"),
      (5L, "l m n"),
      (6L, "mapreduce spark"),
      (7L, "apache hadoop")
    )).toDF("id", "text")

    //make predictions on test documents, cvModel uses the best model found
    crossValidatorModel.transform(testDataSet).select("id", "text", "probability", "prediction").collect().foreach { case Row(id: Long, text: String, prob: Vector, prediction: Double) => println(s"($id, $text) --> prob=$prob, prediction=$prediction") }
    sparkSession.stop()


  }


}
