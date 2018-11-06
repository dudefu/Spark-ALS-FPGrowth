# 1.Spark ALS Demo 从行为数据到评分再到预测

1. ALSDemo.scala 旧的版本
2. ALSRecom.scala 新的版本

上面两个版本，核心代码都是Spark官网上的，区别是输入数据结构不一样。    
https://spark.apache.org/docs/latest/mllib-collaborative-filtering.html#collaborative-filtering


## 字段说明
用户id,商品id,用户行为类型,时间戳，该行为在这一天的次数

（公司的现有数据是只记录某用户一天内的行为次数，没有给每次行为记录一条数据。比如用户一天内浏览了商品3次，只会有一条pv数据，且次数为3。）

## 输入数据
存在 UserBehaviorData.txt 文件中。
```
usfvm1223ds,1,pv,1536940800,3
uhf34sdcfe3,2,pv,1536940800,1
dsfcds2332f,3,pv,1536940800,7
vfcv4356fvf,4,pv,1536940800,1
usfvm1223ds,1,fav,1536940800,1
dvbgf909gbn,1,pv,1536940800,1
dsfcds2332f,2,fav,1536940800,1
vfcv4356fvf,3,pv,1536940800,2
bvdfv487fer,5,pv,1536940800,1
usfvm1223ds,1,buy,1536940800,1
```
## 其他说明
1. 用户浏览次数，统计的是pv（不确定是用pv还是uv，这里代码是用的pv）。
2. 因为ALS算法值接收(userId:Int, itemId:Int, rating:Float)型数据，但是我们的userid是string
3. 这里用了new StringIndexer().fit(dataFrame).transform(dataFrame)来将string类型的userId转成Int。
4. 也可以建立一张中间表（用户id和int型id）。
5. 这里是读取文本文件的数据，还可以用 spark sql来读取数据。
6. 在预测的评分中，会有负分，这是因为用了乔里斯基（Cholesky）分解来解（还有一种是非负最小二乘（NNLS））。可参考：
https://endymecy.gitbooks.io/spark-ml-source-analysis/content/%E6%8E%A8%E8%8D%90/ALS.html

## pom.xml文件
新建maven工程，需要一个正确的pom.xml文件，写代码之前，要保证pom文件的正确。


## 代码输出
### actionDFRDD
```
[usfvm1223ds,1,pv,1536940800,3,0.0]
[uhf34sdcfe3,2,pv,1536940800,1,5.0]
[dsfcds2332f,3,pv,1536940800,7,2.0]
[vfcv4356fvf,4,pv,1536940800,1,1.0]
[usfvm1223ds,1,fav,1536940800,1,0.0]
[dvbgf909gbn,1,pv,1536940800,1,4.0]
[dsfcds2332f,2,fav,1536940800,1,2.0]
[vfcv4356fvf,3,pv,1536940800,2,1.0]
[bvdfv487fer,5,pv,1536940800,1,3.0]
[usfvm1223ds,1,buy,1536940800,1,0.0]
```
### users
```
0
5
2
1
4
3
```
### items
```
1
2
3
4
5
```
### countMap
```
(4-1,0.0)
(1-4,0.0)
(2-3,0.0)
(3-5,0.0)
(1-3,0.0)
(2-2,0.0)
(5-2,0.0)
(0-1,0.0)
```
### countMap
```
Map(4-1 -> 1.0, 1-4 -> 1.0, 2-3 -> 7.0, 3-5 -> 1.0, 1-3 -> 2.0, 2-2 -> 3.0, 5-2 -> 1.0, 0-1 -> 16.0)
```
### 预测结果
```
((1,4),0.9814112189944704)
((1,2),0.8377005876757513)
((4,4),-6.031333549009407E-4)
((1,1),-0.12653905579369407)
((4,2),-0.029346542176744587)
((1,3),1.9999906220495713)
((4,1),0.999953169108017)
((1,5),-0.022793612338590882)
((4,3),-0.06829971098682966)
((3,4),-0.024335730305469788)
((4,5),0.008401785930320831)
((3,2),-0.33523222751490345)
((3,1),1.8090521764939655)
((0,4),-0.009650133678415052)
((3,3),-0.7787258512439097)
((0,2),-0.4695446748279134)
((3,5),0.989916688466003)
((0,1),15.999250705728272)
((5,4),0.11681138852564343)
((0,3),-1.0927953757892745)
((5,2),0.9986909552413586)
((0,5),0.1344285748851333)
((5,1),-0.8203296698520387)
((5,3),2.325605848116709)
((2,4),0.3643438094017086)
((5,5),-0.043520892539204846)
((2,2),3.003574732008076)
((2,1),-2.4651438840544175)
((2,3),6.99508362749611)
((2,5),-0.13042296358488986)
```
### MSE
```
Mean Squared Error = 6.080533743173865E-5
```

# 2.FPGrowth
## 代码输出
### 原始数据集
usfvm1223ds1pv15391008001   
uhf34sdcfe32pv15391008001   
dsfcds2332f6pv15391008002   
vfcv4356fvf8pv15391008001   
usfvm1223ds1pv15391008001   
dvbgf909gbn7pv15391008001   
dsfcds2332f2pv15391008002   
vfcv4356fvf8pv15391008001   
bvdfv487fer2pv15391008003   
usfvm1223ds3pv15391008001   
usfvm1223ds1buy15391008001    
uhf34sdcfe35buy15391008001    
uhf34sdcfe31buy15391008001    
dsfcds2332f3buy15391008002    
vfcv4356fvf5buy15391008001    
usfvm1223ds5buy15391008001    
dvbgf909gbn5buy15391008001    
dsfcds2332f1buy15391008002    
vfcv4356fvf1buy15391008001    
dvbgf909gbn1buy15391008003    
usfvm1223ds2buy15391008001    
usfvm1223ds3buy15391008001    

### 我们只要行为类型是buy的
那么数据只有这些

用户ID    商品ID 行为类型   时间戳   行为次数（浏览了多少次，或者购买了几件，或者收藏了）
usfvm1223ds1buy15391008001    
uhf34sdcfe35buy15391008001    
uhf34sdcfe31buy15391008001    
dsfcds2332f3buy15391008002    
vfcv4356fvf5buy15391008001    
usfvm1223ds5buy15391008001    
dvbgf909gbn5buy15391008001    
dsfcds2332f1buy15391008002    
vfcv4356fvf1buy15391008001    
dvbgf909gbn1buy15391008003    
usfvm1223ds2buy15391008001    
usfvm1223ds3buy15391008001    

在上述数据集中，出现的商品ID分别是：1、5、3、2

### 被购买过的商品数量是：
(buyItem,4)

### 每个用户的购买商品是（购物篮）
CompactBuffer(5, 1)   
CompactBuffer(1, 5, 2, 3)   
CompactBuffer(5, 1)   
CompactBuffer(5, 1)   
CompactBuffer(3, 1)   

### 购物篮数量
(orl-count,5)   

// setMinSupport 支持度support：项集在总项集里面出现的次数。    
// minConfidence 置信度confidence：包含X的项集中，Y出现的概率。u(X,Y)/u(x)。    
	

### 达到最小支持度阈值的项集
[1],5  # 商品1出现5次    
[5],4  # 商品5出现4次    
[5,1],4  # 项集[5,1]出现4次 

### 达到最小置信度阈值的项集
[5]=> [1],1.0  # 商品5相对于商品1的置信度是1.0    
[1]=> [5],0.8  # 商品1相对于商品5的置信度是0.8    
