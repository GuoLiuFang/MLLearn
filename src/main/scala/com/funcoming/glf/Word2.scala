package org.apache.spark.examples.mllib

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
// $example on$
//终于发现问题了，其实是两个不同的包，mllib是老包，功能更全一点。。
import org.apache.spark.mllib.feature.{Word2Vec, Word2VecModel}
// $example off$

object Word2 {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("Word2VecExample").setMaster("local[*]")
    val sc = new SparkContext(conf)

    // $example on$
    val input = sc.textFile("/Users/LiuFangGuo/Downloads/text8").map(line => line.split(" ").toSeq)

    val word2vec = new Word2Vec()

    val model = word2vec.fit(input)

    val synonyms = model.findSynonyms("1", 5)

    for ((synonym, cosineSimilarity) <- synonyms) {
      println(s"$synonym $cosineSimilarity")
    }

    // Save and load model
    model.save(sc, "/Users/LiuFangGuo/Downloads/myModelPath")
    val sameModel = Word2VecModel.load(sc, "/Users/LiuFangGuo/Downloads/myModelPath")
    // $example off$

    sc.stop()
  }
}
