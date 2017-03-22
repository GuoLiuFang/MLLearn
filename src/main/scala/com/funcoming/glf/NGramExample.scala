package com.funcoming.glf

import org.apache.spark.ml.feature.NGram
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession

/**
  * Created by LiuFangGuo on 3/3/17.
  */
object NGramExample {


  def main(args: Array[String]): Unit = {

    val sparkConf = new SparkConf().setMaster("local[*]").setAppName("交叉验证")
    val sparkContext = new SparkContext(sparkConf)
    val sparkSession = SparkSession.builder().appName("是不是session").getOrCreate()

    val wordDataFrame = sparkSession.createDataFrame(Seq(
      (0, Array("Hi", "I", "heard", "about", "Spark")),
      (1, Array("I", "wish", "Java", "could", "use", "case", "classes")),
      (2, Array("Logistic", "regression", "models", "are", "neat"))
    )).toDF("label", "words")

    val ngram = new NGram().setInputCol("words").setOutputCol("ngrams")
    val ngramDataFrame = ngram.transform(wordDataFrame)
    ngramDataFrame.take(3).map(_.getAs[Stream[String]]("ngrams").toList).foreach(println)


  }

}
