package complete.itemcf

import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
import org.jblas.DoubleMatrix

//case class UserAction(user: String, item: Int, action: String, timestamp: String, num: Int)

object itemCFAll {

  val RECOMM_COUNT: Int = 30

  def main(args: Array[String]): Unit = {

    //spark环境配置
    val conf = new SparkConf().setMaster("local[8]").setAppName("itemCF")
    val sc = SparkContext.getOrCreate(conf)
    val spark = SparkSession.builder.getOrCreate()
    sc.setLogLevel("ERROR")

    //读取评分文件
    val data = sc.textFile("file:///D:\\code\\data\\original-data\\10.1-10.11\\ratings\\1")
    data.count()
    //转成Rating
    val ratingRDD = data.map(_.split(",")).map(x => Rating(x(0).toInt,x(1).toInt,x(2).toDouble))
    //模型参数
    val (rank, iterations, lambda) = (50, 5, 0.01)
    //训练ALS模型，利用其生成的productFeatures
    val model = ALS.train(ratingRDD, rank, iterations, lambda)

    // 模型保存于加载
    //    model.save(sc,"D:/code/data/9.15-rate6/model/")
    //    val sameModel = MatrixFactorizationModel.load(sc, "target/tmp/myCollaborativeFilter")

    //获取商品特征向量
    val proFeaRDD = model.productFeatures // RDD[(Int, Array[Double])]
    //相似度计算
    val recommRDD = calSimAndRecomm(model)
    //保存推荐结果
    recommRDD.repartition(1).saveAsTextFile("file:///D:/code/itemCF/itemCFVec/1/")

  }

  /**
    * 相似度计算
    * @param model ALS模型
    */
  private def calSimAndRecomm(model: MatrixFactorizationModel): RDD[(Int,Seq[(Int,Double)])] = {

    // 用于相似度倒叙排序
    object RatingOrder extends Ordering[(Int, Int, Double)] {
      def compare(x: (Int, Int, Double), y: (Int, Int, Double)) = y._3 compare x._3
    }

    // 从模型中提取商品特征，并返回(itemId, factorVector)元组
    val itemVecRDD = model.productFeatures
      .map { case (itemId, factor) =>
        val factorVector = new DoubleMatrix(factor)
        (itemId, factorVector)
      }

//    val minSimilarity = 0.6 // 最小相似度阈值 0.6

    val itemRecomm = itemVecRDD.cartesian(itemVecRDD)
      .filter { case ((itemId1, vector1), (itemId2, vector2)) => itemId1 != itemId2 }//去掉itemID一样的组合
      .map { case ((itemId1, vector1), (itemId2, vector2)) =>
        val sim = cosineSimilarity(vector1, vector2)// 余弦相似度
        (itemId1, itemId2, sim) // 返回元组
      }//.filter(_._3 >= minSimilarity) // 过滤小于阈值的
      .groupBy(p => p._1) // 根据item1分组
      .map { case (itemId: Int, predictions: Iterable[(Int, Int, Double)]) =>
        val recommendations = predictions.toSeq.sorted(RatingOrder)
          .take(RECOMM_COUNT)
          .map(p => (p._2, p._3.toDouble))
        (itemId, recommendations)// 返回格式 (itemId, (p._2, p._3.toDouble))
      }
    itemRecomm
  }

  /**
    * 余弦相似度
    * @param vec1 向量1
    * @param vec2 向量2
    * @return 相似度
    */
  private def cosineSimilarity(vec1: DoubleMatrix, vec2: DoubleMatrix): Double = {
    vec1.dot(vec2) / (vec1.norm2() * vec2.norm2())
  }

}
