package com.funcoming.glf

import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession

/**
  * Created by LiuFangGuo on 3/2/17.
  */
object TfIdfExam {
  def main(args: Array[String]): Unit = {


    val sparkConf = new SparkConf().setMaster("local[*]").setAppName("交叉验证")
    val sparkContext = new SparkContext(sparkConf)
    val sparkSession = SparkSession.builder().appName("是不是session").getOrCreate()


    val sentenceData = sparkSession.createDataFrame(Seq(
      (0, "Hi I heard about Spark"),
      (0, "I wish Java could use case classes"),
      (1, "Logistic regression models are neat")
    )).toDF("label", "sentence")
    val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
    val wordsData = tokenizer.transform(sentenceData)
    val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(20)

    val featuredData = hashingTF.transform(wordsData)

    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    //fit这步仅仅是完成实例化
    val iDFModel = idf.fit(featuredData)
    //transform这一步才是完成了数据的转化。。。
    val rescaledData = iDFModel.transform(featuredData)

    rescaledData.select("features", "label", "sentence").take(3).foreach(println)


  }

}
