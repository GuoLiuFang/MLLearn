package com.funcoming.glf

import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession

/**
  * Created by LiuFangGuo on 3/2/17.
  */
object Word2VecExam {

  // Each row is a bag of words from a sentence or document

  def main(args: Array[String]): Unit = {


    val sparkConf = new SparkConf().setMaster("local[*]").setAppName("交叉验证")
    val sparkContext = new SparkContext(sparkConf)
    val sparkSession = SparkSession.builder().appName("是不是session").getOrCreate()


    val documentDF = sparkSession.createDataFrame(Seq(
      "Hi I heard about Spark".split(" "),
      "I wish Java could use case classes".split(" "),
      "Logistic regression models are neat".split(" ")
    ).map(Tuple1.apply)).toDF("text")

    val word2Vec = new Word2Vec().setInputCol("text").setOutputCol("result").setVectorSize(3).setMinCount(0)
    val model = word2Vec.fit(documentDF)
    val result = model.transform(documentDF)


    result.select("text", "result").take(3).foreach(println)


  }

}
