# coding: utf-8

from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.keras as kr
from jdac.code.model.Model import ModelConfig, CNN
from jdac.code.train.loader import data_convert
from jdac.code.pre_op.doc_fun import get_fact, get_law_list, get_ks, cut, vec


def run():
    data_dir = '../../source/xml'
    dir_list = os.listdir(data_dir)
    input_x1, input_x2, input_ks = [], [], []
    config = ModelConfig()
    model = CNN(config)
    for i in dir_list:
        sub_dir = os.path.join(data_dir, i)
        fact_raw = get_fact(sub_dir)
        statute_raw = get_law_list(sub_dir)
        ks_raw = get_ks(sub_dir)
        after_cut = cut(fact_raw, statute_raw, ks_raw)
        after_vec = vec(after_cut)
        if after_vec.strip() == "":
            continue
        array = after_vec.split('|')
        if len(array) < 5:
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
    input_1 = kr.preprocessing.sequence.pad_sequences(np.array(input_x1), 30)
    input_2 = kr.preprocessing.sequence.pad_sequences(np.array(input_x2), 30)
    ks = np.array(input_ks)
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path="../../result/checkpoints/times-11/con3lstm_att/best_validation")  # 读取保存的模型
    y_pred_cls = np.zeros(shape=len(input_1), dtype=np.int32)  # 保存预测结果
    feed_dict = {
        model.input_x1: input_1,
        model.input_x2: input_2,
        model.input_ks: ks,
        model.keep_prob: 1.0
    }
    y_pred_cls = session.run(model.y_pred_cls, feed_dict=feed_dict)
    return y_pred_cls


res = run()
print(res)
