package com.funcoming.glf

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.ml.linalg.Vector

/**
  * Created by LiuFangGuo on 3/1/17.
  */
object PipeLineExam {
  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setMaster("local[*]").setAppName("看看到底以哪个名字为准")
    val sparkContext = new SparkContext(sparkConf)

    val sparkSession = SparkSession.builder().appName("测试pipleMOdel").getOrCreate()
    val traningDataSet = sparkSession.createDataFrame(Seq(
      (0L, "a b c d e spark", 1.0),
      (1L, "b d", 0.0),
      (2L, "spark f g h", 1.0),
      (3L, "hadoop mapreduce", 0.0)
    )).toDF("id", "text", "label")

    //Configure an ML pipeline , which consists of three stages: tokenizer, hashingTF, and lr.
    val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
    val hashingTF = new HashingTF().setNumFeatures(1000).setInputCol(tokenizer.getOutputCol).setOutputCol("features")
    val logisticRegression = new LogisticRegression().setMaxIter(10).setRegParam(0.01)
    val pipeline = new Pipeline().setStages(Array(tokenizer, hashingTF, logisticRegression))

    val pipelineModel = pipeline.fit(traningDataSet)
    pipelineModel.write.overwrite().save("/Users/LiuFangGuo/Downloads/pipeline-with-fit")
    pipeline.write.overwrite().save("/Users/LiuFangGuo/Downloads/xx-piple")


    val reloadPipelineModel = PipelineModel.load("/Users/LiuFangGuo/Downloads/pipeline-with-fit")
    val testDataSet = sparkSession.createDataFrame(Seq(
      (4L, "spark i j k"),
      (5L, "l m n"),
      (6L, "mapreduce spark"),
      (7L, "apache hadoop")
    )).toDF("id", "text")
    //make predictions on test documents.

    reloadPipelineModel.transform(testDataSet).select("id", "text", "probability", "prediction").collect().foreach { case Row(id: Long, text: String, prob: Vector, predict: Double) => println(s"($id,$text) -----> prob = $prob, predict = $predict") }


    sparkSession.stop()
  }

}
