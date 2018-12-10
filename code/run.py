
#coding="utf-8"




import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import datetime
import pandas as pd
from pandas import rolling_median
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.tsa.stattools as st
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARMA
import math
import random


def trans_gap_time(product_batch):
    #划分label文件中的间隔时间
    #例如  输入是：4.5 16:00 - 00:00 则输出是：tmp_dateTime1='[2018-04-05 16:00]'  tmp_dateTime2='[2018-04-05 24:00]'
    split1 = product_batch.split()
    split2 = split1[0].split(".")
    if int(split2[0]) >= 10:
        if int(split2[1]) >= 10:
            tmp_dateTime = "[2018-"+str(split2[0])+"-"+str(split2[1])
        else:
            tmp_dateTime = "[2018-"+str(split2[0])+"-0"+str(split2[1])
    else:
        if int(split2[1]) >= 10:
            tmp_dateTime = "[2018-0"+str(split2[0])+"-"+str(split2[1])
        else:
            tmp_dateTime = "[2018-0"+str(split2[0])+"-0"+str(split2[1])
    tmp_dateTime1 = tmp_dateTime +" "+ split1[1]+"]"
    if split1[-1] != '00:00':
        tmp_dateTime2 = tmp_dateTime +" "+ split1[-1]+"]"
    else:
        tmp_dateTime2 = tmp_dateTime +" 24:00]"
    return tmp_dateTime1

def writeResult(result,score):
    Now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%m')
    result_file = "../result/"+Now+"-"+str(score)+".csv"
    result.to_csv(result_file,index=False)
    print ("your result saved in %s" %result_file)

def predict_recover(ts):
    ts = np.exp(ts)
    return ts


# ARMA训练模型部分
def ARMA_auto_train_pred(train_data, result, recover=False, write_csv=False):
    # 利用ARMA模型进行自动训练最优值和预测=======
    # recover 是否进行log变化
    #write_csv 是否写入csv
    train_label = train_data[label_columns_list]

    if recover:  # 如果需要进行log变换 就对每列进行变换
        for label in label_columns_list:
            train_label[label] = train_label[label].apply(lambda x: math.log(x))

    # 已经分析这些模型不是非高阶差分平稳数据 所以直接训练最优的p q值====
    st_min_order_list = []
    for label in label_columns_list:
        order = st.arma_order_select_ic(train_label[label], max_ar=6, max_ma=4, ic=['aic', 'bic', 'hqic'])
        st_min_order_list.append(order)
    print("st_min_order_list fished!")

    # 根据'aic', 'bic', 'hqic'指标 分别选取最优值进行模型训练=====
    aic_model_list, bic_model_list, hqic_model_list = [], [], []
    for label, order in zip(label_columns_list, st_min_order_list):
        print("label is %s" % label)
        print("order.aic_min_order is ", order.aic_min_order)
        try:
            model = ARMA(train_label[label], order=order.aic_min_order)
            result_arma = model.fit()
        except:
            print("changed order.aic_min_order to (1,1)")
            model = ARMA(train_label[label], order=(1, 1))
            result_arma = model.fit()
        aic_model_list.append(result_arma)

        print("order.bic_min_order is ", order.bic_min_order)
        try:
            model = ARMA(train_label[label], order=order.bic_min_order)
            result_arma = model.fit()
        except:
            print("changed order.bic_min_order to (1,1)")
            model = ARMA(train_label[label], order=(1, 1))
            result_arma = model.fit()
        bic_model_list.append(result_arma)

        print("orde.hqic_min_order is ", order.hqic_min_order)
        try:
            model = ARMA(train_label[label], order=order.hqic_min_order)
            result_arma = model.fit()
        except:
            print("changed order.hqic_min_order to (1,1)")
            model = ARMA(train_label[label], order=(1, 1))
            result_arma = model.fit()
        hqic_model_list.append(result_arma)

    # 这里是用平均值预测  求得4月预测评价得分
    score = 0
    for label in label_columns_list:
        mean = [train_label[label].mean() for i in range(len(train_label))]
        RMSE = np.sqrt(((mean - train_label[label].values) ** 2).sum() / train_label[label].size)
        print(label, RMSE, math.log10(RMSE))
        score += abs(math.log10(RMSE))
    print("mean socre %f" % (score / 5))

    # 这里是用最优aic值  求得4月预测评价得分
    aic_score = 0
    for label, result_arma in zip(label_columns_list, aic_model_list):
        train_predict = result_arma.predict()
        if recover:
            train_predict = predict_recover(train_predict)  # 还原
        RMSE = np.sqrt(((train_predict - train_label[label]) ** 2).sum() / train_label[label].size)
        print(label, RMSE, math.log10(RMSE))
        aic_score += abs(math.log10(RMSE))
    print("aic socre %f" % (aic_score / 5))

    # 这里是用最优bic值  求得4月预测评价得分
    bic_score = 0
    for label, result_arma in zip(label_columns_list, bic_model_list):
        train_predict = result_arma.predict()
        if recover:
            train_predict = predict_recover(train_predict)  # 还原
        RMSE = np.sqrt(((train_predict - train_label[label]) ** 2).sum() / train_label[label].size)
        print(label, RMSE, math.log10(RMSE))
        bic_score += abs(math.log10(RMSE))
    print("bic socre %f" % (bic_score / 5))

    # 这里是用最优hqic值  求得4月预测评价得分
    hqic_score = 0
    for label, result_arma in zip(label_columns_list, hqic_model_list):
        train_predict = result_arma.predict()
        if recover:
            train_predict = predict_recover(train_predict)  # 还原
        RMSE = np.sqrt(((train_predict - train_label[label]) ** 2).sum() / train_label[label].size)
        print(label, RMSE, math.log10(RMSE))
        hqic_score += abs(math.log10(RMSE))
    print("hqic socre %f" % (hqic_score / 5))

    """

    # 画图分析 分布画出根据'aic', 'bic', 'hqic'指标选择最优模型训练好4月份的结果
    train_label_pred = train_label.copy()
    for i, label in enumerate(label_columns_list):
        train_label[label].plot()

        train_predict_aic = aic_model_list[i].predict()
        train_predict_bic = bic_model_list[i].predict()
        train_predict_hqic = hqic_model_list[i].predict()

        if recover:
            train_predict_aic = predict_recover(train_predict_aic)  # 还原
            train_predict_bic = predict_recover(train_predict_bic)  # 还原
            train_predict_hqic = predict_recover(train_predict_hqic)  # 还原

        train_label_pred[label] = train_predict_aic
        train_label_pred[label].plot()
        train_label_pred[label] = train_predict_bic
        train_label_pred[label].plot()
        train_label_pred[label] = train_predict_hqic
        train_label_pred[label].plot()
        plt.show()
    """

    # 写入预测5月份的结果======
    # 分别写入根据'aic', 'bic', 'hqic'指标选择最优模型预测出来的结果

    # 按照aic指标训练出来的模型线上效果最好
    # 只将线上效果好的结果写入文档

    for label, result_arma in zip(label_columns_list, aic_model_list):
        if recover:
            result[label] = predict_recover(result_arma.forecast(len(result))[0])
        else:
            result[label] = result_arma.forecast(len(result))[0]

    if write_csv:
        if recover:
            writeResult(result, "aic_aram_log_" + str(aic_score / 5))
        else:
            writeResult(result, "aic_aram_" + str(aic_score / 5))
    a_result = result.copy()


    #如果要把其他bic hqic指标的最好结果写入文档 就把下面的两个"""去掉
    """
    result = resul.copy()
    for label, result_arma in zip(label_columns_list, bic_model_list):
        if recover:
            result[label] = predict_recover(result_arma.forecast(len(result))[0])
        else:
            result[label] = result_arma.forecast(len(result))[0]
    if write_csv:        
        if recover:
            writeResult(result, "bic_aram_log_" + str(bic_score / 5))
        else:
            writeResult(result, "bic_aram_" + str(bic_score / 5))
    b_result = result.copy()

    
    for label, result_arma in zip(label_columns_list, hqic_model_list):
        if recover:
            result[label] = predict_recover(result_arma.forecast(len(result))[0])
        else:
            result[label] = result_arma.forecast(len(result))[0]
    if write_csv:  
        if recover:
            writeResult(result, "hqic_aram_log_" + str(hqic_score / 5))
        else:
            writeResult(result, "hqic_aram_" + str(hqic_score / 5))

    hq_result = result.copy()
    
    return a_result, b_result, hq_result
    # """

    #只返回最好的结果
    return a_result


# 噪音调优部分
def reduce_noise(result):
    # 只写入最优结果
    # 模型预测后，可能5月份数据有噪音的加入，使得模型不够完善，预测5月份品质数据不够准确
    # 这里采用了一个假设：每月品质均值波动变化不大，简单的操作就是5月份和4月份的均值一样
    # 具体加噪音调优的思路有三个：
    # （1），在每个时间的数据上，加上4月均值和5月均值差的正态分布数据数  这样就会让5月预测结果加了一个正态分布噪声
    # （2），直接在每个数据上加上均值差  这样相当于在5月份结果上加上了值为均值差的偏执
    # （3），控制预测值加上均值后在4月份最大最小值中间， 相当于数据压缩

    # 线上提交结果显示 直接加偏执效果最好
    r_result = result.copy()  # 思路（1）
    a_result = result.copy()  # 思路（2）
    d_result = result.copy()  # 思路（3）
    for label in result.columns[1:]:
        print(label)
        gap = result[label].mean() - train_data[label].mean()

        max_v, min_v = max(train_data[label]), min(train_data[label])
        label_list = []

        if gap > 0:
            r_result[label] = result[label].apply(lambda x: x - random.uniform(0, gap))  # 加0到平均数差数之间的随机数
            a_result[label] = result[label].apply(lambda x: x + gap)  # 直接加差数
            for v in result[label]:
                if max_v > v - gap > min_v:
                    label_list.append(v - gap)
                else:
                    label_list.append(v)

        else:
            r_result[label] = result[label].apply(lambda x: x + random.uniform(0, gap))
            a_result[label] = result[label].apply(lambda x: x + gap)  # 直接加差数
            for v in result[label]:
                if max_v > v + gap > min_v:
                    label_list.append(v + gap)
                else:
                    label_list.append(v)
        d_result[label] = label_list

    # 只写入最优结果==================
    # writeResult(r_result,"random") #0.37
    writeResult(a_result, "add")  # 0.3052
    # writeResult(d_result,"d") #0.42
    return a_result


# 调优结果之数据分布校验,调整分布
# 未完成，有待提高========
def adjust_distribution(result, adjust_dict={"nitrogen_content": ["uniform", 15.85, (0, 0.5)]}):
    # result 结果DataFrame
    # adjust_dict 调整列名和方法字典 nitrogen_content列需要调整 方法：15.85+random.uniform(0,0.5)
    for columns in adjust_dict.keys():
        if adjust_dict[columns][0] == "uniform":
            result[columns] = [
                adjust_dict[columns][1] + random.uniform(adjust_dict[columns][2][0], adjust_dict[columns][2][1]) \
                for _ in range(len(result))]

    #writeResult(result, "adjust_distribution")
    return result

if __name__ == "__main__":
    # 读取数据=====================
    print("now time is ", datetime.datetime.now())
    print ("load data--------")
    train_data_path = "../data/产品检验报告2018-4-1.csv"
    test_data_path = "../data/产品检验报告2018-5-1-sample.csv"

    train_data = pd.read_csv(open(train_data_path))
    test_data = pd.read_csv(open(test_data_path))

    # product_batch时间转换==================
    train_data["date_index"] = train_data['product_batch'].apply(lambda x: trans_gap_time(x)[1:-1])
    train_data.index = train_data["date_index"]

    test_data["date_index"] = test_data['product_batch'].apply(lambda x: trans_gap_time(x)[1:-1])
    test_data.index = test_data["date_index"]

    # 需要预测的标签
    label_columns_list = list(train_data.columns[2:7])

    # 需要提交填充的数据
    resul = pd.read_csv("../result/sample.csv")
    result = resul.copy()

    print("now time is ", datetime.datetime.now())
    print ("train ARMA model -------")

    a_result = ARMA_auto_train_pred(train_data, result)


    print ("reduce_noise--------- ")
    best_result_1 = reduce_noise(a_result)

    best_result_1.to_csv("../result/result_0.353305.csv", index=False)
    print("now time is ", datetime.datetime.now())

    best_result_2 = adjust_distribution(best_result_1)
    best_result_2.to_csv("../result/result_0.352423_way.csv",index=False)


