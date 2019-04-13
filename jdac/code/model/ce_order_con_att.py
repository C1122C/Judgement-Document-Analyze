# 加入上下文的gate2 in model5
# 添加多级先验知识，并且上一层级得到的[1，d]的score会传入下一层的下一级的计算中，使用的每层计算的权重是两个[d,1]的
import tensorflow as tf


class ModelConfig(object):
    def __init__(self):
        self.EMBEDDING_DIM = 128  # 词向量维度
        self.FACT_LEN = 30  # 事实长度
        self.LAW_LEN = 30  # 法条长度
        self.KS_LEN = 3  # 先验知识长度
        self.LAW_WIN = 8
        self.LAW_STRIDES = 1  # 移动步长

        self.FILTERS = 256
        self.KERNEL_SIZE = 5  # 卷积核尺寸

        self.LAYER_UNITS = 100  # 每层单元数
        self.NUM_CLASS = 2  # 类别数
        self.HIDDEN_DIM = 128

        self.LEARNING_RATE = 0.001  # 学习率
        self.batch_size = 128  # 批处理参数，每次训练跑的数据集样本数
        self.num_epochs = 200  # 跑遍整个数据集的次数
        self.save_per_batch = 10  # 每训练10次保存和打印一次
        self.print_per_batch = 10
        self.dropout_keep_prob = 0.5  # 以0.5的概率去除部分连接防止过拟合


class CNN(object):
    def __init__(self, config):
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
        self.new_x1, pwls = self.gate1(self.input_ks, self.input_x1)
        # new_x1=[ ,30,128],pwls=[ ,30,3,128]是事实与先验知识
        self.new_x1_mean = self.precessf2(self.new_x1)
        # new_x1_mean[128,5]经过处理
        self.new_x2 = self.mirror_gate1(self.new_x1_mean, self.input_x2)
        # 以事实为先验知识筛选出的法条[ ,30,128]
        op = self.conv(self.new_x1, self.new_x2)
        # op1,op2是经过卷积提取出的向量
        self.match(op)

    '''
    s1 = tf.sigmoid(tf.relu(x * w * k_1)) 计算得到关于这个先验知识哪些dim应该被保留,这个weight是[128,1]
    s2 = tf.sigmoid(tf.relu(x * w * [k_2;s1]))
    s3 = tf.sigmoid(tf.relu(x * w * [k_3;s2]))
    new_vector = s1 * x + s2 * x + s3 * x
    '''

    # 结合事实与先验知识
    def gate1(self, ks, input_x):
        with tf.name_scope("gate"):
            weight_1 = tf.Variable(tf.random_normal([30, self.config.EMBEDDING_DIM, 30],
                                                    stddev=0, seed=1), trainable=True, name='w1')
            # 权重初始值随机生成，矩阵128*1,正态分布标准差为0，随机种子为1[128,1]
            weight_2 = tf.Variable(tf.random_normal([30, self.config.EMBEDDING_DIM, 30],
                                                    stddev=0, seed=2), trainable=True, name='w2')
            weight_3 = tf.Variable(tf.random_normal([30, self.config.EMBEDDING_DIM, 30],
                                                    stddev=0, seed=3), trainable=True, name='w3')

            k_1_init, k_2_init, k_3_init = ks[:, 0, :], ks[:, 1, :], ks[:, 2, :]  # 分别切取先验知识的0，1，2行
            k_1 = tf.reshape(tf.keras.backend.repeat_elements(k_1_init, rep=self.config.FACT_LEN, axis=1),
                             shape=[-1, self.config.FACT_LEN, 1, self.config.EMBEDDING_DIM])
            k_2 = tf.reshape(tf.keras.backend.repeat_elements(k_2_init, rep=self.config.FACT_LEN, axis=1),
                             shape=[-1, self.config.FACT_LEN, 1, self.config.EMBEDDING_DIM])
            k_3 = tf.reshape(tf.keras.backend.repeat_elements(k_3_init, rep=self.config.FACT_LEN, axis=1),
                             shape=[-1, self.config.FACT_LEN, 1, self.config.EMBEDDING_DIM])
            '''add content'''
            fun1 = tf.einsum('abd,bde->abe', input_x, weight_1)  # 事实与权重
            fun2 = tf.transpose(fun1, perm=[0, 2, 1])
            fun3 = tf.reduce_mean(fun2, axis=2, keep_dims=True)
            fun3_epd = tf.expand_dims(fun3, axis=2)
            # relu(x)=max(0,x)
            # sigmoid(x)=1/(1+exp(-x))
            # ksw_1代表事实与先验知识1
            '''add content'''
            # [ ,30,1,128]
            ksw_1 = tf.sigmoid(tf.nn.relu(tf.einsum('abcd,abdf->abcf', fun3_epd, k_2)))
            '''add content'''
            fun4 = tf.einsum('abd,bde->abe', input_x, weight_2)
            fun5 = tf.transpose(fun4, perm=[0, 2, 1])
            fun6 = tf.reduce_mean(fun5, axis=2, keep_dims=True)
            fun6_epd = tf.expand_dims(fun6, axis=2)
            # ksw_2代表事实与先验知识2
            '''add content'''
            fun6_epd1 = tf.keras.backend.repeat_elements(fun6_epd, rep=2, axis=3)
            # print('fun6:', fun6_epd1.shape,flush=True)
            # print('k_3:', k_3.shape,flush=True)
            # print('ksw_1:', ksw_1.shape,flush=True)
            ksw_2 = tf.sigmoid(tf.nn.relu(tf.einsum('abcd,abdf->abcf', fun6_epd1, tf.concat([k_3, ksw_1], axis=2))))

            '''add content'''
            fun7 = tf.einsum('abd,bde->abe', input_x, weight_3)
            fun8 = tf.transpose(fun7, perm=[0, 2, 1])
            fun9 = tf.reduce_mean(fun8, axis=2, keep_dims=True)
            fun9_epd = tf.expand_dims(fun9, axis=2)
            fun9_epd1 = tf.keras.backend.repeat_elements(fun9_epd, rep=2, axis=3)
            '''add content'''
            ksw_3 = tf.sigmoid(tf.nn.relu(tf.einsum('abcd,abdf->abcf', fun9_epd1, tf.concat([k_1, ksw_2], axis=2))))
            input_x_epd = tf.expand_dims(input_x, axis=2)
            # 连接
            n_vector_ = (ksw_1 + ksw_2 + ksw_3) * input_x_epd
            n_vector = tf.reshape(n_vector_, shape=[-1, self.config.FACT_LEN, self.config.EMBEDDING_DIM])
            # 形状改为[ ,30,128],ksw_1,2,3在第二个维度拼接[ ,30,3,128]
        return n_vector, tf.concat([ksw_1, ksw_2, ksw_3], axis=2)

    def precessf2(self, input_x):
        with tf.name_scope("FactPrecess"):
            # 将输入的第一维和第三维转置[128,30, ]
            input_x_ = tf.transpose(input_x, perm=[0, 2, 1])
            # input_x_的每行最大的5个数,[0]表示只要数值，不要位置
            # [5,30, ]转换回[ ,30,5]
            input_x_k = tf.transpose((tf.nn.top_k(input_x_, k=5, sorted=False))[0], perm=[0, 2, 1])
            # 对第二维求平均完成降维，变为[ ,5]
            '''add content'''
            input_x_mean = tf.reduce_mean(input_x_k, axis=1)
            return input_x_mean

    '''
    根据事实作为先验知识去过滤法条
    '''
    def mirror_gate1(self, input_x, input_y):
        with tf.name_scope("Fact2Law"):
            # [128,1]
            weight_1 = tf.Variable(tf.random_normal([30, self.config.EMBEDDING_DIM, 30],
                                                    stddev=0, seed=1), trainable=True, name='w1')
            ss_epd = tf.reshape(tf.keras.backend.repeat_elements(input_x, rep=self.config.LAW_LEN, axis=1),
                                shape=[-1, self.config.LAW_LEN, 1, self.config.EMBEDDING_DIM])  # [b,l,1,d]
            # 输入的法条变成[ ,30,1,128]
            law_epd = tf.expand_dims(input_y, axis=2)
            '''add content'''
            fun = tf.einsum('abd,bde->abe', input_y, weight_1)
            fun1 = tf.transpose(fun, perm=[0, 2, 1])
            fun2 = tf.reduce_mean(fun1, axis=2, keep_dims=True)
            fun2_epd = tf.expand_dims(fun2, axis=2)
            # [a,b,c,e] = fun[a,b,c,d]*ss_epd[a,b,d,e]经过relu和sigmoid得到ksw
            '''add content'''
            ksw = tf.sigmoid(tf.nn.relu(tf.einsum('abcd,abde->abce', fun2_epd, ss_epd)))
            # 输出形式[ ,30,128]
            n_vector_ = ksw * law_epd
            n_vector = tf.reshape(n_vector_, shape=tf.shape(input_y))
        return n_vector

    # 生成卷积
    def conv(self, input_x, input_y):
        with tf.name_scope("conv"):
            dot = tf.matmul(input_x, input_y, adjoint_b=True)
            # 对行做attention
            self.beta = tf.nn.softmax(dot, axis=2)
            self.alpha = tf.nn.softmax(dot, axis=1)
            with tf.name_scope('cnn1'):
                conv1 = tf.layers.conv1d(self.beta, self.config.FILTERS, self.config.KERNEL_SIZE, name='conv1')
                gmp1 = tf.reduce_max(conv1, reduction_indices=[1], name='gmp1')

            with tf.name_scope('cnn2'):
                conv2 = tf.layers.conv1d(self.alpha, self.config.FILTERS, self.config.KERNEL_SIZE, name='conv2')
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

            with tf.name_scope("concat"):
                concat = tf.concat([gmp1, gmp2, gmp3, gmp4], 1)
            return concat

    def match(self, op):
        with tf.name_scope("match"):
            fc = tf.layers.dense(inputs=op, units=128, name="fc3_c")
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)  # 根据比例keep_prob输出输入数据，最终返回一个张量
            fc = tf.nn.relu(fc)  # 激活函数，此时fc的维度是hidden_dim

            # 分类器,以fc作为输入，输出大小为2
            self.logits = tf.layers.dense(fc, self.config.NUM_CLASS,
                                          name='fc4_c')  # 将fc从[batch_size,hidden_dim]映射到[batch_size,num_class]输出
            # softmax将向量上的数值映射成概率，argmax选出做大概率所在的索引值
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵，logits和了、labelsd大小都是[batch,2]
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                                    labels=self.input_y)
            # 对logits进行softmax操作后，做交叉墒，输出的是一个向量
            # 将交叉熵向量求和，即可得到交叉熵
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.LEARNING_RATE).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1),
                                    self.y_pred_cls)
            # 由于input_y也是onehot编码，因此，调用tf.argmax(self.input_y)得到的是1所在的下标
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))