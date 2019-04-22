# coding: utf-8

from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.keras as kr
from jdac.code.model.Model import ModelConfig, CNN
from jdac.code.train.loader import data_convert
from jdac.code.pre_op.doc_fun import get_fact, get_law_list, cut, vec

config = ModelConfig()
model = CNN(config)


def run():
    print("IN RIGHT METHOD")
    data_dir = '../../source/xml'
    dir_list = os.listdir(data_dir)
    input_x1, input_x2, input_ks = [], [], []

    d_l = []
    d_num = []
    count = 0
    for i in dir_list:
        sub_dir = os.path.join(data_dir, i)
        d_l.append(sub_dir)
        print("CHECKING " + sub_dir)
        fact_raw = get_fact(sub_dir).split('。')
        # print("FACT LEN:")
        # print(len(fact_raw))
        name_raw, statute_raw, ks_raw = get_law_list(sub_dir)

        after_cut = cut(fact_raw, statute_raw, ks_raw)
        after_vec = vec(after_cut)
        if count == 0:
            d_num.append(len(after_vec))
        else:
            d_num.append(d_num[count - 1] + len(after_vec))

        for j in range(len(after_vec)):
            data = after_vec[j]
            if data.strip() == "":
                continue
            array = data.split('|')
            if len(array) < 3:
                continue
            fact_ls = array[0].split(' ')  # 事实
            statue_ls = array[1].split(' ')  # 法条
            ks_ls = array[2].split(' ')  # 知识
            # 将事实和法条的list都还原为float
            input_x1.append(data_convert(fact_ls))
            input_x2.append(data_convert(statue_ls))
            zs_matrix = data_convert(ks_ls)
            # 添加知识
            input_ks.append(zs_matrix)
    count += 1

    input_1 = kr.preprocessing.sequence.pad_sequences(np.array(input_x1), 30)
    input_2 = kr.preprocessing.sequence.pad_sequences(np.array(input_x2), 30)
    ks = np.array(input_ks)
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path="../../result/checkpoints/times-11/Model/best_validation")  # 读取保存的模型
    y_pred_cls = np.zeros(shape=len(input_1), dtype=np.int32)  # 保存预测结果
    feed_dict = {
        model.input_x1: input_1,
        model.input_x2: input_2,
        model.input_ks: ks,
        model.keep_prob: 1.0
    }
    y_pred_cls = session.run(model.y_pred_cls, feed_dict=feed_dict)
    res = []
    print("文件: " + d_l[0] + "计算结果:")
    k = 0
    for j in range(len(after_vec)):
        if k < len(d_l)-1 and j == d_num[k]:
            k += 1
            print("文件: " + d_l[k] + "计算结果:")
        p_f = fact_raw[int(j / len(name_raw))]
        p_s = name_raw[int(j % len(name_raw))]
        if 1-y_pred_cls[j] < 0.5:
            result = "相关"
        else:
            result = "不相关"
        s = "事实:" + p_f+"和法条:"+p_s+" "+result
        print(s)
        res.append(s)
    return res


res = run()
