package als

import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.ArrayBuffer

case class UserActionIndex(user: String, item: Int, action: String, timestamp: String, num: Int, index: Double)
case class UserAction(user: String, item: Int, action: String, timestamp: String, num: Int)

object ALSDemo {

  val ACTION_PV: String = "pv"
  val ACTION_FAV: String = "fav"
  val ACTION_BUY: String = "buy"

  val RATE_PV: Double = 1
  val RATE_FAV: Double = 3
  val RATE_BUY: Double = 10

  val ITEM_PERCENT = 0.3

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setMaster("local[2]").setAppName("als")
    val sc = SparkContext.getOrCreate(conf)
    sc.setLogLevel("ERROR")

    val spark = SparkSession.builder.getOrCreate()

    //读取行为日志
//    val data_user_action = sc.textFile("file:///ml/data/9.15.txt")
    val data_user_action = sc.textFile("file:///D:/code/scala/demo11/src/main/scala/als/UserBehaviorData.txt")

    //      data_user_action.take(10)
    val userActionsRDD = data_user_action.map(_.split(',') match { case Array(user, item, action, timestamp, num) =>
      UserAction(user.toString, item.toInt, action.toString, timestamp.toString, num.toInt)
    })

    val dataFrame = spark.createDataFrame(userActionsRDD)
    val indexedDf: DataFrame = new StringIndexer().setInputCol("user").setOutputCol("user_code").fit(dataFrame).transform(dataFrame)
    val actionDFRDD = indexedDf.rdd.collect()

//    actionDFRDD foreach println

    //    indexedDf.rdd.saveAsTextFile("file:///D:/code/data/out")

    //构建map，用于存放评分。  key：userId-itemId（字符串拼接），value：评分
    val countMap: scala.collection.mutable.Map[String, Double] = scala.collection.mutable.Map() //新建空map

    //所有用户
    val users = actionDFRDD.map { x =>

      var userId = x(5).toString
      userId = userId.substring(0, userId.length - 2)

      //往map里面添加值
      val key = userId + "-" + x(1) // 用户序号-商品ID
      var value = 0.0 // 评分
      countMap.put(key, value) // map的长度=日志数据的条数
      userId.toInt
    }.distinct

//    users foreach println

    //所有的商品
    val items = actionDFRDD.map(x => x(1).toString.toInt).distinct

//    items foreach println

//    countMap foreach println

    //统计  用户、商品  数量
//    val userRdd = sc.makeRDD(users)
//    val itemRdd = sc.makeRDD(items)

    //遍历行为，累计评分
    indexedDf.rdd.collect().map { x =>

      //index
      val index_temp = x(5).toString
      val index = index_temp.substring(0, index_temp.length - 2)

      val item = x(1) //map的key
      val action = x(2).toString //行为类型
      val num = x(4).toString.toInt //行为次数
      val key = index + "-" + item

      var rate = countMap.getOrElse(key, 0.0) //已有的分值

      //判断行为类型，加上对应的分数
      if (ACTION_PV.equals(action)) {
        rate += RATE_PV * num
      }
      if (ACTION_FAV.equals(action)) {
        rate += RATE_FAV * num
      }
      if (ACTION_BUY.equals(action)) {
        rate += RATE_BUY * num
      }
      countMap.put(key, rate)
    }

//    println(countMap)

    //把评分的map转成RDD
    val ab = ArrayBuffer[Rating]()
    for (a <- countMap) {
      val key = a._1
      val value = a._2

      val arr = key.split("-")
      val userId = arr(0).toInt //
      val itemId = arr(1).toInt

      var myRating = Rating(userId, itemId, value)
      ab += myRating
    }

    val ratingRDD = sc.makeRDD(ab)

    //模型与训练  Build the recommendation model using ALS
    val (rank, iterations, lambda) = (50, 5, 0.01)
    val model = ALS.train(ratingRDD, rank, iterations, lambda)

    // Evaluate the model on rating data
    val usersProducts = ratingRDD.map { case Rating(user, product, rate) =>
      (user, product)
    }
    val predictions =
      model.predict(usersProducts).map { case Rating(user, product, rate) =>
        ((user, product), rate)
      }

//    预测结果
    predictions foreach println
//    ((1,986),2.9963685354079486)
//    ((1,1402),-0.12304401882570264)

    val ratesAndPreds = ratingRDD.map { case Rating(user, product, rate) =>
      ((user, product), rate)
    }.join(predictions)// ((4,1690),(3.0,2.996368536647455))

    val MSE = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
      math.pow(r1 - r2,2) // 差的平方
    }.mean()

    println(s"Mean Squared Error = $MSE")

    val recommRes = model.recommendProductsForUsers(100) // 给每个用户推荐100个商品
//    (3,1815,9.998900479391585)
//    (1,986,2.9963685544722414)

  }
}
