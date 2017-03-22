package com.funcoming.glf

import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by LiuFangGuo on 2/28/17.
  */
object CounterGZ {
  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setMaster("spark://NY-HADOOP-12-151:7777").setAppName("线上计数器")
    //spark   Context是必须有的。。
    val sparkContext = new SparkContext(sparkConf)
    val textFileGzRdd = sparkContext.textFile(args(0)).filter(line => line.contains(""""eventId":"sms_received""""))
    val LineSumCount = textFileGzRdd.count()
    println("文件" + args(0) + "最终的总数为" + LineSumCount)
  }

}
