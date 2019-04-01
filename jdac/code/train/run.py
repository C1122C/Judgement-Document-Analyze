# coding: utf-8

from __future__ import print_function

import os
import time
from datetime import timedelta
import numpy as np
import tensorflow as tf
from sklearn import metrics
from jdac.code.model.ce_order import ModelConfig as m_org
from jdac.code.model.ce_order import CNN as c_org
from jdac.code.model.ce_order_con import ModelConfig as m_con
from jdac.code.model.ce_order_con import CNN as c_con
from jdac.code.model.ce_order_con2 import ModelConfig as m_con2
from jdac.code.model.ce_order_con2 import CNN as c_con2
from jdac.code.model.ce_order_con3 import ModelConfig as m_con3
from jdac.code.model.ce_order_con3 import CNN as c_con3
from jdac.code.model.ce_order_con_weight import ModelConfig as m_conw
from jdac.code.model.ce_order_con_weight import CNN as c_conw
from jdac.code.train.loader import batch_iter, batch_iter_test, data_load

data_dir = '../../source/set_4'
train_path = data_dir+'/train.txt'
val_data_path = data_dir+'/val.txt'
test_path = data_dir + '/test.txt'

t_f = open(train_path, 'r', encoding='utf-8')
v_f = open(val_data_path, 'r', encoding='utf-8')
test_f = open(test_path, 'r', encoding='utf-8')

ks_flag = 3   # kw level
n_number = 1  # n-gram
gate_n = 3
reg = False  # default is false
times = 1
ks_order = '123'
mirror_gate = 1
precessF = '2'
precessL = '1'
topK = 5  # default is 5
last_ks_info = 'False'  # default is true
sing_leuse = '3'
relu = 'False'  # default is true

save_dir = '../../result'
result = '../../record/result.txt'
tm_path = 'times-11'
org_save_path = save_dir+'/checkpoints/'+tm_path+'/original/best_validation'
org_tensor_board_dir = save_dir + '/tensor_board/' + tm_path+'/original/best_validation'
con30_save_path = save_dir + '/checkpoints/' + tm_path + '/con30/best_validation'
con30_tensor_board_dir = save_dir+'/tensor_board/' + tm_path + '/con30/best_validation'
con_v_save_path = save_dir + '/checkpoints/' + tm_path + '/con_v/best_validation'
con_v_tensor_board_dir = save_dir+'/tensor_board/' + tm_path + '/con_v/best_validation'
con_2_save_path = save_dir + '/checkpoints/' + tm_path + '/con_2/best_validation'
con_2_tensor_board_dir = save_dir+'/tensor_board/' + tm_path + '/con_2/best_validation'
con_weight_save_path = save_dir + '/checkpoints/' + tm_path + '/con_weight/best_validation'
con_weight_tensor_board_dir = save_dir+'/tensor_board/' + tm_path + '/con_weight/best_validation'

if not os.path.exists(org_save_path):
    os.makedirs(org_save_path)
if not os.path.exists(org_tensor_board_dir):
    os.makedirs(org_tensor_board_dir)
if not os.path.exists(con30_save_path):
    os.makedirs(con30_save_path)
if not os.path.exists(con30_tensor_board_dir):
    os.makedirs(con30_tensor_board_dir)
if not os.path.exists(con_v_save_path):
    os.makedirs(con_v_save_path)
if not os.path.exists(con_v_tensor_board_dir):
    os.makedirs(con_v_tensor_board_dir)
if not os.path.exists(con_2_save_path):
    os.makedirs(con_2_save_path)
if not os.path.exists(con_2_tensor_board_dir):
    os.makedirs(con_2_tensor_board_dir)
if not os.path.exists(con_weight_save_path):
    os.makedirs(con_weight_save_path)
if not os.path.exists(con_weight_tensor_board_dir):
    os.makedirs(con_weight_tensor_board_dir)

config_o = m_org()
model_o = c_org(config_o)
config_con = m_con()
model_con = c_con(config_con)
config_con2 = m_con2()
model_con2 = c_con2(config_con2)
config_con3 = m_con3()
model_con3 = c_con3(config_con3)
config_w = m_conw()
model_w = c_conw(config_w)

tb_list = [org_tensor_board_dir, con30_tensor_board_dir, con_2_tensor_board_dir,
           con_v_tensor_board_dir, con_weight_tensor_board_dir]
save_list = [org_save_path, con30_save_path, con_2_save_path,
             con_v_save_path, con_weight_save_path]
model_list = [model_o, model_con, model_con2, model_con3, model_w]


# 获取已用时间
def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


# 数据填入
def feed_data(x1_batch, x2_batch, ks_batch, y_batch, keep_prob, mod):
    feed_dict = {
        mod.input_x1: x1_batch,
        mod.input_x2: x2_batch,
        mod.input_ks: ks_batch,
        mod.input_y: y_batch,
        mod.keep_prob: keep_prob
    }
    return feed_dict


# 评估在某一数据集上的准确率和损失，训练用
def evaluate(sess, x1_, x2_, ks_, y_, mod):
    data_len = len(x1_)
    batch_eval = batch_iter(x1_, x2_, ks_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x1_batch, x2_batch, ks_batch, y_batch in batch_eval:
        batch_len = len(x1_batch)
        feed_dict = feed_data(x1_batch, x2_batch, ks_batch, y_batch, 1.0)
        loss, acc = sess.run([mod.loss, mod.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len


# 评估在某一数据集上的准确率和损失，测试用
def evaluate_test(sess, x1_, x2_, ks_, y_, mod):
    data_len = len(x1_)
    batch_eval = batch_iter_test(x1_, x2_, ks_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x1_batch, x2_batch, ks_batch, y_batch in batch_eval:
        batch_len = len(x1_batch)
        feed_dict = feed_data(x1_batch, x2_batch, ks_batch, y_batch, 1.0)
        loss, acc = sess.run([mod.loss, mod.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len


# 训练模型
def train(mod, save_path, tb_path):
    print("Configuring TensorBoard and Saver...")
    # 配置 Tensor_board，重新训练时，请将tensor_board文件夹删除，不然图会覆盖

    # 结果可视化与存储
    tf.summary.scalar("loss", mod.loss)  # 可视化loss
    tf.summary.scalar("accuracy", mod.acc)  # 可视化acc
    merged_summary = tf.summary.merge_all()   # 将所有操作合并输出
    writer = tf.summary.FileWriter(tb_path)  # 将summary data写入磁盘

    # 配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Loading training and validation data...")
    start_time = time.time()
    # 载入训练集
    # 事实、法条、知识、标记
    train_1, train_2, train_ks, train_output = data_load(t_f, mod.config, ks_flag)
    print('train len:', train_1.shape)
    # 载入验证集
    # 事实、法条、知识、标记
    val_1, val_2, val_ks, val_output = data_load(v_f, mod.config, ks_flag)
    print('validation len:', len(val_1))

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # 创建session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练

    flag = False

    for epoch in range(mod.config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(train_1, train_2, train_ks, train_output, mod.config.batch_size)
        for x1_batch, x2_batch, ks_batch, y_batch in batch_train:
            feed_dict = feed_data(x1_batch, x2_batch, ks_batch, y_batch, mod.config.dropout_keep_prob, mod)
            if total_batch % mod.config.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensor_board scalar
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % mod.config.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                feed_dict[mod.keep_prob] = 1.0
                loss_train, acc_train = session.run([mod.loss, mod.acc], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(session, val_1, val_2, val_ks, val_output,mod)  # 验证当前会话中的模型的loss和acc

                if acc_val > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            session.run(mod.optim, feed_dict=feed_dict)  # 运行优化
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  # 跳出循环
        if flag:  # 同上
            break


# 测试模型
def test(dic, mod):
    print("Loading test data for "+dic+"......")
    res = open(result, 'a')
    res.write("Test for "+dic+"\n")
    start_time = time.time()
    x1_test, x2_test, ks_test, y_test = data_load(test_f, mod.config, flag=ks_flag)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=dic)  # 读取保存的模型

    print('Testing...')
    loss_test, acc_test = evaluate_test(session, x1_test, x2_test, ks_test, y_test, mod)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))
    res.write(msg.format(loss_test, acc_test))
    res.write("\n")

    batch_size = 128
    data_len = len(x1_test)
    num_batch = int(data_len / batch_size)

    y_test_cls = np.argmax(y_test, 1)
    y_pred_cls = np.zeros(shape=len(x1_test), dtype=np.int32)  # 保存预测结果

    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            mod.input_x1: x1_test[start_id:end_id],
            mod.input_x2: x2_test[start_id:end_id],
            mod.input_ks: ks_test[start_id:end_id],
            mod.keep_prob: 1.0   # 这个表示测试时不使用dropout对神经元过滤
        }
        # 将所有批次的预测结果都存放在y_pred_cls中
        y_pred_cls[start_id:end_id] = session.run(mod.y_pred_cls, feed_dict=feed_dict)

    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_cls, y_pred_cls, digits=4))  # 直接计算准确率，召回率和f值
    res.write(metrics.classification_report(y_test_cls, y_pred_cls, digits=4))
    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)
    res.write(str(cm))
    res.write("\n")

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    res.write("Time usage:")
    res.write(str(time_dif))
    res.write("\n")
    res.close()
    return y_test_cls, y_pred_cls


for i in range(5):
    train(model_list[i], save_list[i], tb_list[i])
    test(save_list[i], model_list[i])
