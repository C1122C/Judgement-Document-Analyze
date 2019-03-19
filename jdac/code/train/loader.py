# coding: utf-8

import sys
import tensorflow.contrib.keras as kr
import numpy as np


if sys.version_info[0] > 2:
    is_py3 = True
else:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    is_py3 = False


# 去除空字串并将字符串还原为float
def data_convert(vectors):
    ss_ls = list(filter(lambda x: x.strip() != '', vectors))
    n_vector = [list(map(float, list(filter(lambda x: x.strip() != '', ss.split('//'))))) for ss in ss_ls]
    return n_vector


# 从文件读取数据
def data_load(data_f, config, flag=3):
    input_x1, input_x2, input_ks, input_y = [], [], [], []
    lines = data_f.read().split('\n')
    for i in range(len(lines)):
        line = lines[i]
        # print('index:', i)
        if line.strip() == "":
            continue

        array = line.split('|')
        if len(array) < 5:
            continue
        ss_ls = array[1].split(' ')  # 事实
        ftzw_ls = array[2].split(' ')  # 法条
        label = int(array[3].strip())  # 标签1或0
        zs_ls = array[4].split(' ')  # 知识
        # 将事实和法条的list都还原为float
        input_x1.append(data_convert(ss_ls))
        input_x2.append(data_convert(ftzw_ls))
        if label == 0:
            input_y.append([1, 0])
        else:
            input_y.append([0, 1])
        if flag == 1:  # mean vectors of words of 'jie'
            zs_matrix = np.mean(data_convert(zs_ls), axis=0)
        else:
            zs_matrix = data_convert(zs_ls)
        # 添加知识
        input_ks.append(zs_matrix)

    print('input_x.shape:', len(input_ks[9]))
    # 将事实和法条都改为模型固定的长度
    train_1 = kr.preprocessing.sequence.pad_sequences(np.array(input_x1), config.FACT_LEN)
    train_2 = kr.preprocessing.sequence.pad_sequences(np.array(input_x2), config.LAW_LEN)
    train_ks = np.array(input_ks)

    return train_1, train_2, train_ks, np.array(input_y)


# compute context
def add_context(input_x):
    batch_size = input_x.shape[0]
    new_input_x = []
    for _ in range(batch_size):
        sample = []
        for i in range(1, input_x.shape[1]):
            ctx = (input_x[_, i - 1] + input_x[_, i + 1]) / 2
            sample.append(ctx)
        new_input_x.append(sample)
    return np.array(new_input_x)


# 生成批次数据（事实，法条，知识，标签）
def batch_iter(x1, x2, ks, y, batch_size=128):
    data_len = len(x1)
    num_batch = int(data_len / batch_size)  # 计算可以生成几批

    indices = np.random.permutation(np.arange(data_len))  # 洗牌
    x1_shuffle = x1[indices]
    x2_shuffle = x2[indices]
    ks_shuffle = ks[indices]
    y_shuffle = y[indices]

    # 一次一个批次地返回打乱的数据
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x1_shuffle[start_id:end_id], x2_shuffle[start_id:end_id], ks_shuffle[start_id:end_id], y_shuffle[start_id: end_id]


# 测试数据批次处理
def batch_iter_test(x1, x2, ks, y, batch_size=128):
    data_len = len(x1)
    num_batch = int(data_len / batch_size)  # 批次数

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x1[start_id:end_id], x2[start_id:end_id], ks[start_id:end_id], y[start_id:end_id]

