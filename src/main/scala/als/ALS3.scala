package als

import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.{ArrayBuffer, Map}

case class UserActionIndex(user: String, item: Int, action: String, timestamp: String, num: Int, index: Double)
case class UserAction(user: String, item: Int, action: String, timestamp: String, num: Int)

object ScalaDemo {

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

//    println("XIN")
//    actionDFRDD foreach println

    //    indexedDf.rdd.saveAsTextFile("file:///D:/code/data/out")

    //构建map，用于存放评分。  key：userId-itemId（字符串拼接），value：评分
    var countMap: Map[String, Double] = scala.collection.mutable.Map() //新建空map

    //所有用户
    val users = actionDFRDD.map { x =>

      var userId = x(5).toString
      userId = userId.substring(0, userId.length - 2)

      //往map里面添加值
      var key = userId + "-" + x(1) // 用户序号-商品ID
    var value = 0.0 // 评分
      countMap.put(key, value) // map的长度=日志数据的条数
      userId.toInt
    }.distinct

//    println("XIN")
//    users foreach println

    //所有的商品
    val items = actionDFRDD.map(x => x(1).toString.toInt).distinct

//    println("XIN")
//    items foreach println

//    println("XIN")
//    countMap foreach println

    //用户-商品 笛卡尔积
    val userRdd = sc.makeRDD(users)
    val itemRdd = sc.makeRDD(items)

    val userItems = userRdd.cartesian(itemRdd)


    val userItemsRel = userItems.collect() //获取所有值

    //遍历行为，累计评分
    indexedDf.rdd.collect().map { x =>

      //index
      var index_temp = x(5).toString
      var index = index_temp.substring(0, index_temp.length - 2)

      var item = x(1) //map的key
    var action = x(2).toString //行为类型
    var num = x(4).toString.toInt //行为次数

      var key = index + "-" + item

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


//    println("XIN")
//    println(countMap)

    //把评分的map转成RDD
    val ab = ArrayBuffer[Rating]()
    for (a <- countMap) {
      var key = a._1
      var value = a._2

      var arr = key.split("-")
      var userId = arr(0).substring(0, arr.length - 1).toInt
      var itemId = arr(1).toInt

      var myRating = new Rating(userId, itemId, value)
      ab += myRating
    }

    var ratingRDD = sc.makeRDD(ab)

    //模型与训练  Build the recommendation model using ALS
    val (rank, iterations, lambda) = (50, 5, 0.01)
    val model = ALS.train(ratingRDD, rank, iterations, lambda)

    // Evaluate the model on rating data
    val usersProducts = ratingRDD.map { case Rating(user, product, rate) =>
      (user, product)
    }
    val predictions =
      model.predict(userItems).map { case Rating(user, product, rate) =>
        ((user, product), rate)
      }

//    预测结果
//    println("XIN")
    predictions foreach println
//    ((1,986),2.9963685354079486)
//    ((1,1402),-0.12304401882570264)
//    ((4,986),0.21814732076066246)
//    ((1,1690),0.218147393238044)
//    ((4,1402),0.029785994599512974)

    val ratesAndPreds = ratingRDD.map { case Rating(user, product, rate) =>
      ((user, product), rate)
    }.join(predictions)// ((4,1690),(3.0,2.996368536647455))

    val MSE = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
      val err = (r1 - r2)
      err * err
    }.mean()

//    println("XIN")
//    println(s"Mean Squared Error = $MSE")

    // 查看某个用户对某个商品的评分
//    va1 userPred1 = model.predict(1,986)
//    println(userPred1)

    // TODO 发现推荐指数很高的商品
    // 取出评分较高的商品

    val sortedPred = predictions.sortBy(x => x._2, ascending = false ).collect

//    // 确定取前多少，不写逻辑
//    val itemCount = itemRdd.count()// 商品总个数
//    val needCount = itemCount * ITEM_PERCENT  // 取多少个
//    val needItem = sortedPred.take(needCount.toInt)

      for ( x <- sortedPred){
        var (user, item ) = x._1
        var rate = x._2
        println(user,item,rate)
      }

//    (3,1815,9.998900479391585)
//    (1,986,2.9963685544722414)
//    (4,1690,2.9963685417944315)
//    (0,3239,0.9899167003756413)
//    (2,1402,0.9899166983248724)
//    (1,1815,0.6242709202296586)
//    (2,986,0.4484347664725926)
//    (3,986,0.18900726007135238)
//    (4,1815,0.17580693760420862)
//    (1,1402,0.16150188358139383)
//    (3,1690,0.05322818935220758)
//    (2,1690,0.024253716199680198)
//    (4,1402,0.00873487109478574)
//    (3,3239,-0.013201018620516432)
//    (2,3239,-0.026183289429579454)
//    (0,1402,-0.026183297102424216)
//    (3,1402,-0.07675419294350438)
//    (1,3239,-0.08918152721477333)
//    (4,3239,-0.11087113058168335)
//    (0,1815,-0.1210665014602571)
//    (1,1690,-0.23261303426205585)
//    (4,986,-0.23261305139232302)
//    (0,986,-0.24762625007715783)
//    (0,1690,-0.3078507312089259)
//    (2,1815,-0.7039120866471468)

  }
}
