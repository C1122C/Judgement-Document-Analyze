#!/usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf


class ModelConfig(object):
    def __init__(self):
        self.EMBEDDING_DIM = 128  # 词向量维度
        self.FACT_LEN = 30  # 事实长度
        self.LAW_LEN = 30  # 法条长度
        self.NUM_CLASS = 2  # 类别数
        self.NUM_LAYERS = 2
        self.HIDDEN_DIM = 128
        self.LEARNING_RATE = 0.001  # 学习率
        self.batch_size = 128  # 批处理参数，每次训练跑的数据集样本数
        self.num_epochs = 200  # 跑遍整个数据集的次数
        self.save_per_batch = 10  # 每训练10次保存和打印一次
        self.print_per_batch = 10
        self.dropout_keep_prob = 0.8  # 以0.5的概率去除部分连接防止过拟合


class TextRNN(object):
    """文本分类，RNN模型"""
    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x1 = tf.placeholder(tf.float32, [None, self.config.FACT_LEN, self.config.EMBEDDING_DIM],
                                       name='input_x1')
        self.input_x2 = tf.placeholder(tf.float32, [None, self.config.LAW_LEN, self.config.EMBEDDING_DIM],
                                       name='input_x2')
        self.input_y = tf.placeholder(tf.int32, [None, self.config.NUM_CLASS], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.rnn()

    def rnn(self):
        """rnn模型"""

        def lstm_cell():   # lstm核
            return tf.contrib.rnn.BasicLSTMCell(self.config.HIDDEN_DIM, forget_bias=1.0, state_is_tuple=True)

        def dropout():  # 为每一个rnn核后面加一个dropout层
            return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=self.keep_prob)

        with tf.name_scope("bi_lstm"):
            with tf.variable_scope("bi-lstm1"):
                cell_fw = [dropout() for _ in range(self.config.NUM_LAYERS)]
                cell_bw = [dropout() for _ in range(self.config.NUM_LAYERS)]

                cell_fw = tf.contrib.rnn.MultiRNNCell(cell_fw, state_is_tuple=True)
                cell_bw = tf.contrib.rnn.MultiRNNCell(cell_bw, state_is_tuple=True)

                init_state_fw = cell_fw.zero_state(batch_size=self.config.batch_size, dtype=tf.float32)
                init_state_bw = cell_bw.zero_state(batch_size=self.config.batch_size, dtype=tf.float32)

                _outputs_1, state_1 = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw,
                                                                      inputs=self.input_x1,
                                                                      initial_state_fw=init_state_fw,
                                                                      initial_state_bw=init_state_bw)

            with tf.variable_scope("bi-lstm2"):
                cell_fw = [dropout() for _ in range(self.config.NUM_LAYERS)]
                cell_bw = [dropout() for _ in range(self.config.NUM_LAYERS)]

                cell_fw = tf.contrib.rnn.MultiRNNCell(cell_fw, state_is_tuple=True)
                cell_bw = tf.contrib.rnn.MultiRNNCell(cell_bw, state_is_tuple=True)

                init_state_fw = cell_fw.zero_state(batch_size=self.config.batch_size, dtype=tf.float32)
                init_state_bw = cell_bw.zero_state(batch_size=self.config.batch_size, dtype=tf.float32)

                _outputs_2, state_2 = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw,
                                                                      inputs=self.input_x2,
                                                                      initial_state_fw=init_state_fw,
                                                                      initial_state_bw=init_state_bw)

        with tf.name_scope("sum"):
            with tf.variable_scope("input1_sum"):
                # shape:[batch_size,max_time,output_size]
                output_fw = _outputs_1[0]
                output_bw = _outputs_1[1]
                # shape:[batch_size,output_size]
                last_fw = output_fw[:, -1, :]
                last_bw = output_bw[:, -1, :]
                # shape:[batch_size,output_size]
                last1 = tf.add(last_fw, last_bw)/2
            with tf.variable_scope("input2_sum"):
                # shape:[batch_size,max_time,output_size]
                output_fw = _outputs_2[0]
                output_bw = _outputs_2[1]
                # shape:[batch_size,output_size]
                last_fw = output_fw[:, -1, :]
                last_bw = output_bw[:, -1, :]
                # shape:[ batch_size,output_size]
                last2 = tf.add(last_fw, last_bw) / 2

        with tf.name_scope('concat'):
            result = tf.concat([last1, last2], 1)

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            # w*input+b,其中可以在此方法中指定w,b的初始值，或者通过tf.get_varable指定
            fc = tf.layers.dense(result, self.config.HIDDEN_DIM, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            # 根据比例keep_prob输出输入数据，最终返回一个张量
            fc = tf.nn.relu(fc)
            # 激活函数，此时fc的维度是hidden_dim

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.NUM_CLASS, name='fc2')
            # 将fc从[batch_size,hidden_dim]映射到[batch_size,num_class]输出
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)
            # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            # 对logits进行softmax操作后，做交叉墒，输出的是一个向量
            self.loss = tf.reduce_mean(cross_entropy)
            # 将交叉熵向量求和，即可得到交叉熵
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.LEARNING_RATE).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            # 由于input_y也是onehot编码，因此，调用tf.argmax(self.input_y)得到的是1所在的下表
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
