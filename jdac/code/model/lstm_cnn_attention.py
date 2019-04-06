#!/usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf


class ModelConfig(object):
    """RNN配置参数"""

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
        self.rnn = 'lstm'  # lstm 或 gru
        self.num_filters = 256  # 卷积核数目
        self.kernel_size = 5  # 卷积核尺寸


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
            return tf.contrib.rnn.BasicLSTMCell(self.config.HIDDEN_DIM, state_is_tuple=True)

        def gru_cell():  # gru核
            return tf.contrib.rnn.GRUCell(self.config.HIDDEN_DIM)

        def dropout():  # 为每一个rnn核后面加一个dropout层
            if self.config.rnn == 'lstm':
                cell = lstm_cell()
            else:
                cell = gru_cell()
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

        with tf.name_scope("rnn1"):
            # 多层rnn网络
            with tf.variable_scope("rnn1v"):
                cells_1 = [dropout() for _ in range(self.config.NUM_LAYERS)]
                rnn_cell_1 = tf.contrib.rnn.MultiRNNCell(cells_1, state_is_tuple=True)
                _outputs_1, state_1 = tf.nn.dynamic_rnn(cell=rnn_cell_1, inputs=self.input_x1, dtype=tf.float32,
                                                        time_major=False)
                # time_major等于 false代表输入和输出的格式是[batch_size, max_time, depth]

        with tf.name_scope("rnn2"):
            # 多层rnn网络
            with tf.variable_scope("rnn2v"):
                cells_2 = [dropout() for _ in range(self.config.NUM_LAYERS)]
                rnn_cell_2 = tf.contrib.rnn.MultiRNNCell(cells_2, state_is_tuple=True)
                _outputs_2, state_2 = tf.nn.dynamic_rnn(cell=rnn_cell_2, inputs=self.input_x2, dtype=tf.float32,
                                                        time_major=False)
                # time_major等于 false代表输入和输出的格式是[batch_size, max_time, depth]

        with tf.name_scope("attention"):
            dot = tf.matmul(_outputs_1, _outputs_2, adjoint_b=True)
            # 对行做attention
            self.beta = tf.nn.softmax(dot, axis=2)
            self.alpha = tf.nn.softmax(dot, axis=1)

        with tf.name_scope('cnn1'):
            conv1 = tf.layers.conv1d(self.beta, self.config.num_filters, self.config.kernel_size, name='conv1')
            gmp1 = tf.reduce_max(conv1, reduction_indices=[1], name='gmp1')

        with tf.name_scope('cnn2'):
            conv2 = tf.layers.conv1d(self.alpha, self.config.num_filters, self.config.kernel_size, name='conv2')
            gmp2 = tf.reduce_max(conv2, reduction_indices=[1], name='gmp2')

        with tf.name_scope("cnn3"):
            # CNN layer
            with tf.variable_scope("cnn-var1"):
                conv3 = tf.layers.conv1d(self.input_x1, self.config.num_filters, self.config.kernel_size,
                                         name='conv3')  # (?,26,256)
                gmp3 = tf.reduce_max(conv3, reduction_indices=[1], name='gmp3')  # (?,256)

        with tf.name_scope("cnn4"):
            # CNN layer
            with tf.variable_scope("cnn-var2"):
                conv4 = tf.layers.conv1d(self.input_x2, self.config.num_filters, self.config.kernel_size,
                                         name='conv4')  # (?,46,256)
                gmp4 = tf.reduce_max(conv4, reduction_indices=[1], name='gmp4')  # (?,256)

        with tf.name_scope("concat"):
            concat = tf.concat([gmp1, gmp2, gmp3, gmp4], 1)

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(concat, self.config.HIDDEN_DIM, name='fc1')
            # w*input+b,其中可以在此方法中指定w,b的初始值，或者通过tf.get_varable指定
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
