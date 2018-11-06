package com.xin

import java.text.SimpleDateFormat
import java.util.Date

import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.{ArrayBuffer, HashMap, ListBuffer}

/**
  *
  * ALS离线预测模型
  */
object ALSRecom {

  val conf = new SparkConf().setAppName("ALSCode2Recomm")
   .setMaster("local[*]")
  val sc = SparkContext.getOrCreate(conf)
  sc.setLogLevel("ERROR")

  val startTime = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date)
  //当天时间，格式20181010，用于输出目录名字
  val today = new SimpleDateFormat("yyyyMMdd").format(new Date)
  //行为类型
  val ACTION_PV: String = "pv"
  val ACTION_FAV: String = "fav"
  val ACTION_BUY: String = "buy"
  var usersCount = ""
  var itemsCount = ""
  //每种行为的评分
  var RATING_PV: Double = 1
  var RATING_FAV: Double = 3
  var RATING_BUY: Double = 10
  //输入输出目录
  var train = ""
  //打分文件、推荐结果输出目录
  var output = ""
  //推荐个数
  var topN = 100
  var rank = 15
  var iterations = 15
  var lambda = 0.01

  def main(args: Array[String]): Unit = {

    if (args.length > 1) {
      readPara(args)
    } else {
      println("default parameter")
      train = "file:///D:/code/data/original-data/20181026-28/30.txt"
      output = "file:///D:/code/data/original-data/20181026-28/recomm/" + today + "/" + System.currentTimeMillis()
      printPara
    }

    //读取行为日志
    val data_user_action = sc.textFile(train)
    val actionRDD = data_user_action.map(_.split("\\001") match { case Array(code, item, action, timestamp, num, user) =>
      UserActionCode(code.toInt, item.toInt, action.toString, timestamp.toString, num.toInt, user.toString)
    })

    //构建map，用于存放评分。  key：userId-itemId（字符串拼接），value：评分
    val countMap: HashMap[String, Double] = HashMap() //新建空map
    val userCodeMap: HashMap[Int, String] = HashMap() //新建空map

    //所有用户
    val users = actionRDD.collect().map {
      x =>
        val userCode = x.userCode
        val key = x.userCode + "-" + x.item // 用户序号-商品ID
      var value = 0.0 // 评分
        //往map里面添加值
        countMap.put(key, value) // map的长度=日志数据的条数
        userCodeMap.put(userCode, x.userString)
        userCode
    }.distinct
    //所有的商品
    val items = actionRDD.map(_.item).distinct
    usersCount = users.size.toString
    itemsCount = items.count().toString

    //遍历行为，累计评分
    val actions = actionRDD.collect
    actions foreach {
      x =>
        val userCode = x.userCode
        //map的key
        val item = x.item //map的key
      val action = x.action //行为类型
      val num = x.num //行为次数
      val key = userCode + "-" + item //map的key

        var rate = countMap.getOrElse(key, 0.0) //已有的分值

        //判断行为类型，加上对应的分数
        if (ACTION_PV.equals(action)) {
          rate += RATING_PV * num
        }
        if (ACTION_FAV.equals(action)) {
          rate += RATING_FAV * num
        }
        if (ACTION_BUY.equals(action)) {
          rate += RATING_BUY * num
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
      val userId = arr(0).toInt
      val itemId = arr(1).toInt

      var myRating = Rating(userId, itemId, value)
      ab += myRating
    }

    val ratingRDD = sc.makeRDD(ab).cache()
    //保存打分文件
    ratingRDD.map {
      x =>
        x.user + "," + x.product + "," + x.rating
    }.repartition(1).saveAsTextFile(output + "ratings")

    //模型与训练  Build the recommendation model using ALS
    val model = ALS.train(ratingRDD, rank, iterations, lambda)
    //    val model = ALS.trainImplicit(ratingRDD, rank, iterations, lambda)

    // Evaluate the model on rating data
    val usersProducts = ratingRDD.map { case Rating(user, product, rate) =>
      (user, product)
    }
    val predictions =
      model.predict(usersProducts).map { case Rating(user, product, rate) =>
        ((user, product), rate)
      }
    val ratesAndPreds = ratingRDD.map { case Rating(user, product, rate) =>
      ((user, product), rate)
    }.join(predictions)
    val MSE = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
      val err = (r1 - r2)
      err * err
    }.mean()
    println(s"Mean Squared Error = $MSE")

    val recommRes = model.recommendProductsForUsers(topN) //


    //(997217,[Lorg.apache.spark.mllib.recommendation.Rating;@14c8ed8)
    val resList: ListBuffer[Rating] = new ListBuffer()
    val recom = recommRes.collect()
    recom foreach {
      x =>
        val user = x._1
        val items = x._2
        items foreach {
          y =>
            val item = y.product
            val rating = y.rating
            resList.append(Rating(user, item, rating))
        }
    }

    //保存推荐结果
    //1.保存到磁盘
    val recomRDD = sc.makeRDD(resList)
    recomRDD.map {
      x =>
        x.user + "," + x.product + "," + x.rating
    }.saveAsTextFile(output + "recomm")

    val map: HashMap[String, String] = new HashMap[String, String]()
    recomRDD.groupBy(_.user).collect().foreach {
      x =>
        val code = x._1
        val lists = x._2

        val userStr = userCodeMap.getOrElse(code, "")

        println(userStr)
        var i = 1
        lists foreach {
          y =>
            val item = y.product
            val rating = y.rating
            i += 1
        }
    }

    savePara(MSE)
  }

  private def savePara(MSE: Double) = {

    val list: ListBuffer[(String, String)] = new ListBuffer[(String, String)]
    list.append(("users count", usersCount))
    list.append(("items count", itemsCount))
    list.append(("train", train))
    list.append(("output", output))
    list.append(("topN", topN.toString))
    list.append(("RATING_PV", RATING_PV.toString))
    list.append(("RATING_FAV", RATING_FAV.toString))
    list.append(("RATING_BUY", RATING_BUY.toString))
    list.append(("rank", rank.toString))
    list.append(("iterations", iterations.toString))
    list.append(("lambda", lambda.toString))
    list.append(("Mean Squared Error", MSE.toString))
    list.append(("startTime", startTime))
    list.append(("endTime", new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date)))
    sc.makeRDD(list).repartition(1).saveAsTextFile(output + "mse")
  }

  private def readPara(args: Array[String]) = {
    println("start parameter")
    train = args(0)
    output = args(1) + today + "/"
    topN = args(2).toInt
    RATING_PV = args(3).toDouble
    RATING_FAV = args(4).toDouble
    RATING_BUY = args(5).toDouble
    rank = args(6).toInt
    iterations = args(7).toInt
    lambda = args(8).toDouble
    printPara
  }

  private def printPara = {
    println("train", train)
    println("output", output)
    println("topN", topN)
    println("RATING_PV", RATING_PV)
    println("RATING_FAV", RATING_FAV)
    println("RATING_BUY", RATING_BUY)
    println("rank", rank)
    println("iterations", iterations)
    println("lambda", lambda)
  }

}


