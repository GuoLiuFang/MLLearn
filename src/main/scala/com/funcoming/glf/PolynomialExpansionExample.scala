package com.funcoming.glf

import org.apache.spark.ml.feature.PolynomialExpansion
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession

/**
  * Created by LiuFangGuo on 3/3/17.
  */
object PolynomialExpansionExample {


  def main(args: Array[String]): Unit = {



    val sparkConf = new SparkConf().setMaster("local[*]").setAppName("交叉验证")
    val sparkContext = new SparkContext(sparkConf)
    val sparkSession = SparkSession.builder().appName("是不是session").getOrCreate()


    val data = Array(
      Vectors.dense(-2.0, 2.3),
      Vectors.dense(0.0, 0.0),
      Vectors.dense(0.6, -1.1)
    )
    val df = sparkSession.createDataFrame(data.map(Tuple1.apply)).toDF("features")
    val polynomialExpansion = new PolynomialExpansion()
      .setInputCol("features")
      .setOutputCol("polyFeatures")
      .setDegree(3)
    val polyDF = polynomialExpansion.transform(df)
    polyDF.select("polyFeatures").take(3).foreach(println)


  }

}
