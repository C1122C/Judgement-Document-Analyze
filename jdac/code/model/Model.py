# 加入上下文的gate2 in model5
# 添加多级先验知识，并且上一层级得到的[1，d]的score会传入下一层的下一级的计算中，使用的每层计算的权重是两个[d,1]的
import tensorflow as tf


class ModelConfig(object):
    def __init__(self):
        self.EMBEDDING_DIM = 128  # 词向量维度
        self.FACT_LEN = 30  # 事实长度
        self.LAW_LEN = 30  # 法条长度
        self.KS_LEN = 3  # 先验知识长度

        self.FILTERS = 256
        self.KERNEL_SIZE = 5  # 卷积核尺寸

        self.NUM_CLASS = 2  # 类别数
        self.NUM_LAYERS = 2
        self.HIDDEN_DIM = 128

        self.LEARNING_RATE = 0.001  # 学习率
        self.batch_size = 128  # 批处理参数，每次训练跑的数据集样本数
        self.num_epochs = 200  # 跑遍整个数据集的次数
        self.save_per_batch = 10  # 每训练10次保存和打印一次
        self.print_per_batch = 10
        self.dropout_keep_prob = 0.5  # 以0.5的概率去除部分连接防止过拟合


class CNN(object):
    def __init__(self, config):
        print("IN RIGHT MODEL")
        self.config = config
        self.input_x1 = tf.placeholder(tf.float32, [None, self.config.FACT_LEN, self.config.EMBEDDING_DIM],
                                       name='input_x1')  # 输入1：事实[ ,30,128]
        self.input_x2 = tf.placeholder(tf.float32, [None, self.config.LAW_LEN, self.config.EMBEDDING_DIM],
                                       name='input_x2')  # 输入2：法条[ ,30,128]
        self.input_ks = tf.placeholder(tf.float32, [None, self.config.KS_LEN, self.config.EMBEDDING_DIM],
                                       name="input_ks")  # 输入3：先验知识[ ,3,128]
        self.input_y = tf.placeholder(tf.int32, [None, self.config.NUM_CLASS],
                                      name='input_y')  # 输入4：分类数[ ,2]
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.cnn()
        return

    def cnn(self):
        # 获得第一、二个通道的事实特征提取结果
        new_x1 = self.gate_con(self.input_ks, self.input_x1)
        new_x1_word = self.gate_word(self.input_ks, self.input_x1)

        # 事实特征提取结果经过维度转换，作为法条特征提取的先验知识
        new_x1_mean = self.process_fact(new_x1)
        new_x1_mean_word = self.process_fact(new_x1_word)

        # 获得第一、二个通道的法条特征提取结果
        new_x2 = self.mirror_gate_con(new_x1_mean, self.input_x2)
        new_x2_word = self.mirror_gate_word(new_x1_mean_word, self.input_x2)

        # 获得第三个通道的事实和法条特征提取结果
        x1_bi, x2_bi = self.rnn()
        # 使用注意力机制并完成特征向量连接
        op = self.conv(new_x1_word, new_x2_word, new_x1, new_x2, x1_bi, x2_bi)

        # 全连接层分类并度量准确率和损失
        self.match(op)

    # 用与法条相关的先验知识过滤事实内容
    def gate_con(self, ks, input_x):
        with tf.name_scope("gate_con"):
            # 随机生成权重初始值
            weight_1 = tf.Variable(tf.random_normal([30, self.config.EMBEDDING_DIM, 30],
                                                    stddev=0, seed=1), trainable=True, name='w1')
            weight_2 = tf.Variable(tf.random_normal([30, self.config.EMBEDDING_DIM, 30],
                                                    stddev=0, seed=2), trainable=True, name='w2')
            weight_3 = tf.Variable(tf.random_normal([30, self.config.EMBEDDING_DIM, 30],
                                                    stddev=0, seed=3), trainable=True, name='w3')
            # 抽取出分级的原始先验知识
            k_1_init, k_2_init, k_3_init = ks[:, 0, :], ks[:, 1, :], ks[:, 2, :]
            k_1 = tf.reshape(tf.keras.backend.repeat_elements(k_1_init, rep=self.config.FACT_LEN, axis=1),
                             shape=[-1, self.config.FACT_LEN, 1, self.config.EMBEDDING_DIM])
            k_2 = tf.reshape(tf.keras.backend.repeat_elements(k_2_init, rep=self.config.FACT_LEN, axis=1),
                             shape=[-1, self.config.FACT_LEN, 1, self.config.EMBEDDING_DIM])
            k_3 = tf.reshape(tf.keras.backend.repeat_elements(k_3_init, rep=self.config.FACT_LEN, axis=1),
                             shape=[-1, self.config.FACT_LEN, 1, self.config.EMBEDDING_DIM])
            # 事实进行结合上下文的线性变化
            fun1 = tf.einsum('abd,bde->abe', input_x, weight_1)
            fun2 = tf.transpose(fun1, perm=[0, 2, 1])
            fun3 = tf.reduce_mean(fun2, axis=2, keep_dims=True)
            fun3_epd = tf.expand_dims(fun3, axis=2)
            # 第二级先验知识作用于事实
            # [ ,30,1,128]
            ksw_1 = tf.sigmoid(tf.nn.relu(tf.einsum('abcd,abdf->abcf', fun3_epd, k_2)))

            # 事实进行结合上下文的线性变化
            fun4 = tf.einsum('abd,bde->abe', input_x, weight_2)
            fun5 = tf.transpose(fun4, perm=[0, 2, 1])
            fun6 = tf.reduce_mean(fun5, axis=2, keep_dims=True)
            fun6_epd = tf.expand_dims(fun6, axis=2)
            fun6_epd1 = tf.keras.backend.repeat_elements(fun6_epd, rep=2, axis=3)
            # print('fun6:', fun6_epd1.shape,flush=True)
            # 第三级先验知识和上文得到的作用结果一起作为先验知识作用于事实
            ksw_2 = tf.sigmoid(tf.nn.relu(tf.einsum('abcd,abdf->abcf', fun6_epd1, tf.concat([k_3, ksw_1], axis=2))))

            # 事实进行结合上下文的线性变化
            fun7 = tf.einsum('abd,bde->abe', input_x, weight_3)
            fun8 = tf.transpose(fun7, perm=[0, 2, 1])
            fun9 = tf.reduce_mean(fun8, axis=2, keep_dims=True)
            fun9_epd = tf.expand_dims(fun9, axis=2)
            fun9_epd1 = tf.keras.backend.repeat_elements(fun9_epd, rep=2, axis=3)
            # 第一级先验知识和上文得到的作用结果一起作为先验知识作用于事实
            ksw_3 = tf.sigmoid(tf.nn.relu(tf.einsum('abcd,abdf->abcf', fun9_epd1, tf.concat([k_1, ksw_2], axis=2))))
            input_x_epd = tf.expand_dims(input_x, axis=2)
            # 连接
            n_vector_ = (ksw_1 + ksw_2 + ksw_3) * input_x_epd
            n_vector = tf.reshape(n_vector_, shape=[-1, self.config.FACT_LEN, self.config.EMBEDDING_DIM])

        return n_vector

    def gate_word(self, ks, input_x):
        with tf.name_scope("gate_word"):
            weight_1 = tf.Variable(tf.random_normal([self.config.EMBEDDING_DIM, 1],
                                                    stddev=0, seed=1), trainable=True, name='w1')
            weight_2 = tf.Variable(tf.random_normal([self.config.EMBEDDING_DIM, 2],
                                                    stddev=0, seed=2), trainable=True, name='w2')
            weight_3 = tf.Variable(tf.random_normal([self.config.EMBEDDING_DIM, 2],
                                                    stddev=0, seed=3), trainable=True, name='w3')

            k_1_init, k_2_init, k_3_init = ks[:, 0, :], ks[:, 1, :], ks[:, 2, :]  # 分别切取先验知识的0，1，2行
            k_1 = tf.reshape(tf.keras.backend.repeat_elements(k_1_init, rep=self.config.FACT_LEN, axis=1),
                             shape=[-1, self.config.FACT_LEN, 1, self.config.EMBEDDING_DIM])  # [ ,30,1,128]变形
            k_2 = tf.reshape(tf.keras.backend.repeat_elements(k_2_init, rep=self.config.FACT_LEN, axis=1),
                             shape=[-1, self.config.FACT_LEN, 1, self.config.EMBEDDING_DIM])
            k_3 = tf.reshape(tf.keras.backend.repeat_elements(k_3_init, rep=self.config.FACT_LEN, axis=1),
                             shape=[-1, self.config.FACT_LEN, 1, self.config.EMBEDDING_DIM])
            input_x_epd = tf.expand_dims(input_x, axis=2)
            # fun1[a,b,c,e]=input_x_epd[a,b,c,d]*weight_1[d,e]
            fun1 = tf.einsum('abcd,de->abce', input_x_epd, weight_1)  # 事实与权重

            ksw_1 = tf.sigmoid(tf.nn.relu(tf.einsum('abcd,abdf->abcf', fun1, k_2)))

            # fun2[ ,30,1,2]
            fun2 = tf.einsum('abcd,de->abce', input_x_epd, weight_2)

            # print('fun2:', fun2.shape, flush=True)
            # print('k_3:', k_3.shape, flush=True)
            # print('ksw_1:', ksw_1.shape, flush=True)
            ksw_2 = tf.sigmoid(tf.nn.relu(tf.einsum('abcd,abdf->abcf', fun2, tf.concat([k_3, ksw_1], axis=2))))

            # fun3[a,b,c,e]=input_x_epd[a,b,c,d]*weight_3[d,e]
            fun3 = tf.einsum('abcd,de->abce', input_x_epd, weight_3)

            ksw_3 = tf.sigmoid(tf.nn.relu(tf.einsum('abcd,abdf->abcf', fun3, tf.concat([k_1, ksw_2], axis=2))))

            # 连接
            n_vector_ = (ksw_1 + ksw_2 + ksw_3) * input_x_epd
            n_vector = tf.reshape(n_vector_, shape=[-1, self.config.FACT_LEN, self.config.EMBEDDING_DIM])

        return n_vector

    def process_fact(self, input_x):
        with tf.name_scope("FactProcess"):
            input_x_ = tf.transpose(input_x, perm=[0, 2, 1])
            # input_x_的每行最大的5个数,[0]表示只要数值，不要位置
            input_x_k = tf.transpose((tf.nn.top_k(input_x_, k=5, sorted=False))[0], perm=[0, 2, 1])
            input_x_mean = tf.reduce_mean(input_x_k, axis=1)
            return input_x_mean

    # 根据事实作为先验知识过滤法条
    def mirror_gate_con(self, input_x, input_y):
        with tf.name_scope("Fact2Law"):
            weight_1 = tf.Variable(tf.random_normal([30, self.config.EMBEDDING_DIM, 30],
                                                    stddev=0, seed=1), trainable=True, name='w1')
            ss_epd = tf.reshape(tf.keras.backend.repeat_elements(input_x, rep=self.config.LAW_LEN, axis=1),
                                shape=[-1, self.config.LAW_LEN, 1, self.config.EMBEDDING_DIM])

            law_epd = tf.expand_dims(input_y, axis=2)
            # 线性变化
            fun = tf.einsum('abd,bde->abe', input_y, weight_1)
            fun1 = tf.transpose(fun, perm=[0, 2, 1])
            fun2 = tf.reduce_mean(fun1, axis=2, keep_dims=True)
            fun2_epd = tf.expand_dims(fun2, axis=2)
            # 以事实为先验知识过滤法条内容
            ksw = tf.sigmoid(tf.nn.relu(tf.einsum('abcd,abde->abce', fun2_epd, ss_epd)))

            n_vector_ = ksw * law_epd
            n_vector = tf.reshape(n_vector_, shape=tf.shape(input_y))
        return n_vector

    def mirror_gate_word(self, input_x, input_y):
        with tf.name_scope("Fact2Law_word"):
            weight_1 = tf.Variable(tf.random_normal([self.config.EMBEDDING_DIM, 1],
                                                    stddev=0, seed=1), trainable=True, name='w1')
            ss_epd = tf.reshape(tf.keras.backend.repeat_elements(input_x, rep=self.config.LAW_LEN, axis=1),
                                shape=[-1, self.config.LAW_LEN, 1, self.config.EMBEDDING_DIM])  # [b,l,1,d]
            # 输入的法条变成[ ,30,1,128]
            law_epd = tf.expand_dims(input_y, axis=2)
            # fun[a,b,c,e] = law_epd[a,b,c,d]*weight_1[d,e]
            fun = tf.einsum('abcd,de->abce', law_epd, weight_1)
            # [a,b,c,e] = fun[a,b,c,d]*ss_epd[a,b,d,e]经过relu和sigmoid得到ksw
            ksw = tf.sigmoid(tf.nn.relu(tf.einsum('abcd,abde->abce', fun, ss_epd)))
            # 输出形式[ ,30,128]
            n_vector_ = ksw * law_epd
            n_vector = tf.reshape(n_vector_, shape=tf.shape(input_y))
        return n_vector

    def lstm_cell(self):  # lstm核
        return tf.contrib.rnn.BasicLSTMCell(self.config.HIDDEN_DIM, state_is_tuple=True)

    def dropout(self):  # 为每一个rnn核后面加一个dropout层
        cell = self.lstm_cell()
        return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

    def rnn(self):
        with tf.name_scope("rnn1"):
            # 多层rnn网络
            with tf.variable_scope("rnn1v"):
                cells_1 = [self.dropout() for _ in range(self.config.NUM_LAYERS)]
                rnn_cell_1 = tf.contrib.rnn.MultiRNNCell(cells_1, state_is_tuple=True)
                _outputs_1, state_1 = tf.nn.dynamic_rnn(cell=rnn_cell_1, inputs=self.input_x1, dtype=tf.float32,
                                                        time_major=False)
                # time_major等于 false代表输入和输出的格式是[batch_size, max_time, depth]

        with tf.name_scope("rnn2"):
            # 多层rnn网络
            with tf.variable_scope("rnn2v"):
                cells_2 = [self.dropout() for _ in range(self.config.NUM_LAYERS)]
                rnn_cell_2 = tf.contrib.rnn.MultiRNNCell(cells_2, state_is_tuple=True)
                _outputs_2, state_2 = tf.nn.dynamic_rnn(cell=rnn_cell_2, inputs=self.input_x2, dtype=tf.float32,
                                                        time_major=False)
                # time_major等于 false代表输入和输出的格式是[batch_size, max_time, depth]

        return _outputs_1, _outputs_2

    # 生成卷积
    # bi通道shape[128,30,256],剩下两个[128,30,128]
    def conv(self, x_word, y_word, input_x, input_y, x_bi, y_bi):
        with tf.name_scope("attention"):
            # 对三个通道提取的三组特征做attention
            dot1 = tf.matmul(x_bi, y_bi, adjoint_b=True)
            self.beta1 = tf.nn.softmax(dot1, axis=2)
            self.alpha1 = tf.nn.softmax(dot1, axis=1)

            dot2 = tf.matmul(x_word, y_word, adjoint_b=True)
            self.beta2 = tf.nn.softmax(dot2, axis=2)
            self.alpha2 = tf.nn.softmax(dot2, axis=1)

            dot3 = tf.matmul(input_x, input_y, adjoint_b=True)
            self.beta3 = tf.nn.softmax(dot3, axis=2)
            self.alpha3 = tf.nn.softmax(dot3, axis=1)

        with tf.name_scope('cnn1'):
            conv1 = tf.layers.conv1d(self.beta1, self.config.FILTERS, self.config.KERNEL_SIZE, name='conv1')
            gmp1 = tf.reduce_max(conv1, reduction_indices=[1], name='gmp1')

        with tf.name_scope('cnn2'):
            conv2 = tf.layers.conv1d(self.alpha1, self.config.FILTERS, self.config.KERNEL_SIZE, name='conv2')
            gmp2 = tf.reduce_max(conv2, reduction_indices=[1], name='gmp2')

        with tf.name_scope("cnn3"):
            # CNN layer
            with tf.variable_scope("cnn-var1"):
                conv3 = tf.layers.conv1d(self.input_x1, self.config.FILTERS, self.config.KERNEL_SIZE,
                                         name='conv3')  # (?,26,256)
                gmp3 = tf.reduce_max(conv3, reduction_indices=[1], name='gmp3')  # (?,256)

        with tf.name_scope("cnn4"):
            # CNN layer
            with tf.variable_scope("cnn-var2"):
                conv4 = tf.layers.conv1d(self.input_x2, self.config.FILTERS, self.config.KERNEL_SIZE,
                                         name='conv4')  # (?,46,256)
                gmp4 = tf.reduce_max(conv4, reduction_indices=[1], name='gmp4')  # (?,256)
        with tf.name_scope("cnn5"):
            conv5 = tf.layers.conv1d(self.beta2, filters=self.config.FILTERS, kernel_size=self.config.KERNEL_SIZE,
                                     name='conv5')
            gmp5 = tf.reduce_max(conv5, reduction_indices=[1], name='gmp5')
        with tf.name_scope("cnn6"):
            conv6 = tf.layers.conv1d(self.alpha2, filters=self.config.FILTERS, kernel_size=self.config.KERNEL_SIZE,
                                     name='conv6')
            gmp6 = tf.reduce_max(conv6, reduction_indices=[1], name='gmp6')
        with tf.name_scope("cnn7"):
            conv7 = tf.layers.conv1d(self.beta3, filters=self.config.FILTERS, kernel_size=self.config.KERNEL_SIZE,
                                     name='conv7')
            gmp7 = tf.reduce_max(conv7, reduction_indices=[1], name='gmp7')
        with tf.name_scope("cnn8"):
            conv8 = tf.layers.conv1d(self.alpha3, filters=self.config.FILTERS, kernel_size=self.config.KERNEL_SIZE,
                                     name='conv8')
            gmp8 = tf.reduce_max(conv8, reduction_indices=[1], name='gmp8')
        with tf.name_scope("concat"):
            concat = tf.concat([gmp1, gmp2, gmp5, gmp6, gmp7, gmp8, gmp3, gmp4], 1)
        return concat

    # op size[128,256]
    def match(self, op):
        with tf.name_scope("match_c3"):
            fc = tf.layers.dense(inputs=op, units=self.config.HIDDEN_DIM, name="fc3_3")
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.NUM_CLASS,
                                          name='fc4_3')
            # softmax将向量上的数值映射成概率，argmax选出做大概率所在的索引值
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)

        with tf.name_scope("optimize_c3"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                                    labels=self.input_y)
            # 交叉熵
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.LEARNING_RATE).minimize(self.loss)

        with tf.name_scope("accuracy_c3"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1),
                                    self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
