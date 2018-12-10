# <center>[化工产品品质智能预测算法](http://www.hfdatacity.com/common/cmpt/%E5%8C%96%E5%B7%A5%E4%BA%A7%E5%93%81%E5%93%81%E8%B4%A8%E6%99%BA%E8%83%BD%E9%A2%84%E6%B5%8B_%E7%AB%9E%E8%B5%9B%E4%BF%A1%E6%81%AF.html)</center>   


赛题比较简单，运气好，拿了第一，请大佬们轻拍多指教！ 

本赛题目数据跨时2个月,第一个月的数据用于训练,第二个月的数据用于测试。数据可以描述为3类：
1. 生产参数记录表。(36个测点的传感器数据)
2. 产品检测结果。(需要预测的结果)
3. 生产工艺流程。(生产参数和结果的相互关系)    

###  <font color=OrangeRed>评分方法：</font>
![在这里插入图片描述](https://img-blog.csdnimg.cn/2018120922060499.png)  

其中，〖RMSE〗_i表示第i个检测项目的RMSE分数，共5个检测项目，分别是磷含量%、 氮含量%、总养分%、含水量%和粒度%。score越小越好。
## 思路(一)：尝试提取生产数据当特征建立回归模型
- 1,将生产参数记录表中的数据当成训练特征,产品检测结果当成标签数据,建立回归模型。
- 2,根据生产结果中的时间段,在生产参数记录表中特征在规定时间的平均值、和、检测次数作为特征。
- 3,建立LR、BayesianRidge、SVR、GradientBoostingRegressor等模型训练预测。
结论：LR的效果超出其他模型很多,说明数据具有线性关系。再加上用这种提取生产参数数据的方式建立模型,对于特征提取的要求高,难度较大,所以建议利用时序回归模型。	<font color=OrangeRed>主要是线上评分太低,所以改用时序模型。</font>

## 思路(一)优缺点
### 优点：
- 1,	抗噪音。因为(1)生产参数特征较多且全面,(2)根据生产参数预测检测结果,类似端到端的训练模型,能捕捉特征反映的结果。
- 2,	查生产过程问题。可以得到每个生产参数的权重 ,如果产品质量出现问题,可以很快很好定位到哪个生产参数记录表出现问题,排查改善。
### 缺点：
- 1,	生产参数特征不好提取。有缺失不规律,很难提取关键特征。
- 2,	模型准确率低。正因为无法提取较好的特征,所以模型一般很差。

## 思路(二)：直接利用检测结果建立ARMA模型
- 1,	将检测结果按照“product_batch”列为时序,建立时间序列数据
- 2,	对数据进行平稳性检测：基本上是一阶差分或本身平稳。
- 3,	随机性检测,原数据是非白噪声,所以可以直接用原数据建立ARMA模型。
- 4,	确定ARMA阶数,这里采用两种方法,一是画出ACF、PACF观察p,q值,锁定。
- 5,	ARMA模型参数max_ar=6,max_ma=4。而是根据锁定的参数训练模型依据aic、bic、hqic指标搜索最优p,q值。
(训练结果见下组图,蓝色是4月真实值,其他颜色是依据aic bic hqic最优指标预测4月)	
    
    
nitrogen_content 建模效果： 
    
    
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181208231659467.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTAxOTk0NDE=,size_16,color_FFFFFF,t_70)    

    
    
particle_size 建模效果：    
    
    
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181208231812805.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTAxOTk0NDE=,size_16,color_FFFFFF,t_70)    

    
    
phosphorus_content 建模效果：    
    
    
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181208231919195.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTAxOTk0NDE=,size_16,color_FFFFFF,t_70)    

    
    
total_nutrient 建模效果：     
    
    
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181208231941927.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTAxOTk0NDE=,size_16,color_FFFFFF,t_70)    
    
    
    
               
water_content 建模效果： 
    
    
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181208232004145.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTAxOTk0NDE=,size_16,color_FFFFFF,t_70)    
    
    
    
- 6,	根据最优p,q值训练模型,线下评价。
- 7,	噪音调优部分：
模型预测后,可能5月份数据有噪音的加入,使得模型不够完善,预测5月份品质数据不够准确。
这里采用了一个假设：每月品质均值波动变化不大,简单的操作就是5月份和4月份的均值一样
具体加噪音调优的思路有三个：  
  (1), 在每个时间的数据上,加上4月均值和5月均值差的正态分布数据数,这样就会让5月预测结果加了一个正态分布噪声。    
  (2),直接在每个数据上加上均值差,这样相当于在5月份结果上加上了值为均值差的偏置。    
  (3),控制预测值加上均值后在4月份最大最小值中间,相当于数据压缩。  
  <font color=OrangeRed>线上提交结果显示：设置(arma_order_select_ic)选择最优参数max_ar=6,max_ma=4,选择出aic指标下最优的p,q值,训练出的ARMA,在6噪音调优中加偏置(2)效果最好,B榜分数0.353305。</font>
- 8,	各品牌检测分布调优
在分析5月份预测结果分布和4月份真是分布发现,“nitrogen_content”围绕均值15.85呈现(0,0.5)之间随机分布情况,所以用15.85+(0,0.5)随机分布来预测“nitrogen_content”指标,<font color=OrangeRed>B榜分数0.352423。</font>
 - <font color=OrangeRed>9,	未完成部分:</font>    
(1)	原始数据的异常值检测,去除后训练模型可能效果更好。   
(2)	选择ARMA模型的p,q值可能可以优化,因为只是认为主管设置max_ar=6,max_ma=4。    
(3)	原始数据log变换或其他变换后再训练可能效果更好。

# 思路(二)：优缺点
### 优点：
- 1,	对于短期稳定数据预测准确度高。
- 2,	建模方便容易。
### 缺点：
- 1,	对于长期趋势难预测。
- 2,	异常值对于模型干扰严重。
- 3,	难以准确预测产品质量哪天会出问题,风控能力小。	

 <font color=OrangeRed size=6>参考文献</font>   
 [1]: [AR(I)MA时间序列建模过程](https://www.jianshu.com/p/cced6617b423)     
 [2]: [化工产品品质智能预测大赛](http://www.hfdatacity.com/common/cmpt/%E5%8C%96%E5%B7%A5%E4%BA%A7%E5%93%81%E5%93%81%E8%B4%A8%E6%99%BA%E8%83%BD%E9%A2%84%E6%B5%8B_%E7%AB%9E%E8%B5%9B%E4%BF%A1%E6%81%AF.html)    
 [3]: [Mygithub](https://github.com/CraigZhangCq)
