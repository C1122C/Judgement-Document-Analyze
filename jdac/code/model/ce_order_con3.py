# 加入上下文的gate2 in model5
# 添加多级先验知识，并且上一层级得到的[1，d]的score会传入下一层的下一级的计算中，使用的每层计算的权重是两个[d,1]的
import tensorflow as tf
import numpy as np



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

        self.LEARNING_RATE = 0.001  # 学习率
        self.batch_size = 128  # 批处理参数，每次训练跑的数据集样本数
        self.num_epochs = 200  # 跑遍整个数据集的次数
        self.save_per_batch = 10  # 每训练10次保存和打印一次
        self.print_per_batch = 10
        self.dropout_keep_prob = 0.5  # 以0.5的概率去除部分连接防止过拟合


class CNN(object):
    def __init__(self, config):
        self.config = config
        self.input_x1 = tf.placeholder(tf.float32, [128, self.config.FACT_LEN, self.config.EMBEDDING_DIM],
                                       name='input_x1')  # 输入1：事实[ ,30,128]
        self.input_x2 = tf.placeholder(tf.float32, [128, self.config.LAW_LEN, self.config.EMBEDDING_DIM],
                                       name='input_x2')  # 输入2：法条[ ,30,128]
        self.input_ks = tf.placeholder(tf.float32, [128, self.config.KS_LEN, self.config.EMBEDDING_DIM],
                                       name="input_ks")  # 输入3：先验知识[ ,3,128]
        self.input_y = tf.placeholder(tf.int32, [128, self.config.NUM_CLASS],
                                      name='input_y')  # 输入4：分类数[ ,2]
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.cnn()
        return

    def cnn(self):
        self.new_x1, pwls = self.gate1(self.input_ks, self.input_x1)
        self.new_x1_word, pwls_word = self.gate1_word(self.input_ks, self.input_x1)
        # new_x1=[ ,30,128],pwls=[ ,30,3,128]是事实与先验知识
        self.new_x1_mean = self.precessf2(self.new_x1)
        self.new_x1_mean_word = self.precessf2(self.new_x1_word)
        # new_x1_mean[128,5]经过处理
        self.new_x2 = self.mirror_gate1(self.new_x1_mean, self.input_x2)
        self.new_x2_word = self.mirror_gate1_word(self.new_x1_mean_word, self.input_x2)
        # 以事实为先验知识筛选出的法条[ ,30,128]
        # shape[128,30,256]
        self.x1_bi, self.x2_bi = self.bi_gru(self.input_x1, self.input_x2)
        op1, op2 = self.conv(self.new_x1_word, self.new_x2_word, self.new_x1, self.new_x2, self.x1_bi, self.x2_bi)
        #op1_word, op2_word = self.conv(self.new_x1_word, self.new_x2_word)
        # op1,op2是经过卷积提取出的向量

        self.match(op1, op2)

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

    def gate1_word(self, ks, input_x):
        with tf.name_scope("gate_word"):
            weight_1 = tf.Variable(tf.random_normal([self.config.EMBEDDING_DIM, 1],
                                                    stddev=0, seed=1), trainable=True, name='w1')
            # 权重初始值随机生成，矩阵128*1,正态分布标准差为0，随机种子为1[128,1]
            weight_2 = tf.Variable(tf.random_normal([self.config.EMBEDDING_DIM, 2],
                                                    stddev=0, seed=2), trainable=True, name='w2')
            weight_3 = tf.Variable(tf.random_normal([self.config.EMBEDDING_DIM, 2],
                                                    stddev=0, seed=3), trainable=True, name='w3')

            k_1_init, k_2_init, k_3_init = ks[:, 0, :], ks[:, 1, :], ks[:, 2, :]  # 分别切取先验知识的0，1，2行
            k_1 = tf.reshape(tf.keras.backend.repeat_elements(k_1_init, rep=self.config.FACT_LEN, axis=1),  # 变为[ ,128*30]
                             shape=[-1, self.config.FACT_LEN, 1, self.config.EMBEDDING_DIM])  # [ ,30,1,128]变形
            k_2 = tf.reshape(tf.keras.backend.repeat_elements(k_2_init, rep=self.config.FACT_LEN, axis=1),
                             shape=[-1, self.config.FACT_LEN, 1, self.config.EMBEDDING_DIM])
            k_3 = tf.reshape(tf.keras.backend.repeat_elements(k_3_init, rep=self.config.FACT_LEN, axis=1),
                             shape=[-1, self.config.FACT_LEN, 1, self.config.EMBEDDING_DIM])
            input_x_epd = tf.expand_dims(input_x, axis=2)  # [ ,30,1,128]
            # fun1[a,b,c,e]=input_x_epd[a,b,c,d]*weight_1[d,e]
            fun1 = tf.einsum('abcd,de->abce', input_x_epd, weight_1)  # 事实与权重
            # [a,b,c,f]=fun1[a,b,c,d]*k_2[a,b,d,f]
            # relu(x)=max(0,x)
            # sigmoid(x)=1/(1+exp(-x))
            # ksw_1代表事实与先验知识1
            ksw_1 = tf.sigmoid(tf.nn.relu(tf.einsum('abcd,abdf->abcf', fun1, k_2)))

            # fun2[ ,30,1,2]
            fun2 = tf.einsum('abcd,de->abce', input_x_epd, weight_2)
            # ksw_2代表事实与先验知识2
            print('fun2:', fun2.shape, flush=True)
            print('k_3:', k_3.shape, flush=True)
            print('ksw_1:', ksw_1.shape, flush=True)
            ksw_2 = tf.sigmoid(tf.nn.relu(tf.einsum('abcd,abdf->abcf', fun2, tf.concat([k_3,ksw_1],axis=2)))) # [batch,l,d]

            # fun3[a,b,c,e]=input_x_epd[a,b,c,d]*weight_3[d,e]
            fun3 = tf.einsum('abcd,de->abce',input_x_epd , weight_3)
            # ksw_3代表事实与先验知识3
            ksw_3 = tf.sigmoid(tf.nn.relu(tf.einsum('abcd,abdf->abcf', fun3, tf.concat([k_1,ksw_2],axis=2)))) # [batch,l,d]

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

    def mirror_gate1_word(self, input_x, input_y):
        with tf.name_scope("Fact2Law_word"):
            # [128,1]
            weight_1 = tf.Variable(tf.random_normal([self.config.EMBEDDING_DIM, 1],
                                                    stddev=0, seed=1), trainable=True, name='w1')
            # 输入的事实由[128,5]变为[128,5*30]再变为[5,30,1,128]
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

    def bi_gru(self, fact, law):
        # 优：考虑参数设置
        sentence2vec = BiGRU(n_in=128, n_hidden=128, n_out=200, batch_size=128)
        # 优：考虑return_list是否为true

        # 优：考虑return_list是否为true
        # shape[128,30,256]
        q_emb = sentence2vec(fact, return_list=True)
        r_emb = sentence2vec(law, return_list=True)
        return q_emb, r_emb

    # 生成卷积
    # bi通道shape[128,30,256],剩下两个[128,30,128]
    def conv(self, x_word, y_word, input_x, input_y, x_bi, y_bi):
        with tf.name_scope("conv"):
            weight_1 = tf.Variable([0.3, 0.3, 0.3], dtype="float32", trainable=True, name='w1')
            weight_2 = tf.Variable([0.3, 0.3, 0.3], dtype="float32", trainable=True, name='w2')
            # 对事实输入生成卷积，过滤器个数256，一维卷积窗口大小5
            # 输入30*128，卷积尺寸5*128，得到26*1的向量，因为有256个过滤器，所以共有256个26*1向量
            conv1 = tf.layers.conv1d(input_x, filters=self.config.FILTERS, kernel_size=self.config.KERNEL_SIZE,
                                     name='conv3')
            # 在第二维的26个值中选出最大的，得到一个有256个值的向量
            # op1[128,256] con30fact
            op_con30fact = tf.reduce_max(conv1, reduction_indices=[1], name='gmp1')

            # 对法条输入生成卷积，过滤器个数256，一维卷积窗口大小5
            conv2 = tf.layers.conv1d(input_y, filters=self.config.FILTERS, kernel_size=self.config.KERNEL_SIZE,
                                     name='conv4')
            # op2[128,256] con30law
            op_con30law = tf.reduce_max(conv2, reduction_indices=[1], name='gmp2')
            conv3 = tf.layers.conv1d(x_bi, filters=self.config.FILTERS, kernel_size=self.config.KERNEL_SIZE,
                                     name='conv5')
            # op3[128,256] bi_fact
            op_bi_fact = tf.reduce_max(conv3, reduction_indices=[1], name='gmp3')
            conv4 = tf.layers.conv1d(y_bi, filters=self.config.FILTERS, kernel_size=self.config.KERNEL_SIZE,
                                     name='conv6')
            # op4[128,256] bi_law
            op_bi_law = tf.reduce_max(conv4, reduction_indices=[1], name='gmp4')
            conv5 = tf.layers.conv1d(x_word, filters=self.config.FILTERS, kernel_size=self.config.KERNEL_SIZE,
                                     name='conv7')
            # op5[128,256] word_fact
            op_word_fact = tf.reduce_max(conv5, reduction_indices=[1], name='gmp5')
            conv6 = tf.layers.conv1d(y_word, filters=self.config.FILTERS, kernel_size=self.config.KERNEL_SIZE,
                                     name='conv8')
            # op6[128,256] word_law
            op_word_law = tf.reduce_max(conv6, reduction_indices=[1], name='gmp6')

            op1 = weight_1[0] * op_word_fact + weight_1[1] * op_con30fact + weight_1[2] * op_bi_fact
            op2 = weight_2[0] * op_word_law + weight_2[1] * op_con30law + weight_2[2] * op_bi_law
            return op1, op2

    # op size[128,256]
    def match(self, op1, op2):
        with tf.name_scope("match"):
            h = tf.concat([op1, op2], axis=1)  # [batch,FILTERS*2]
            # 全连接层，输出大小为100，使用偏置项，参与训练
            fc = tf.layers.dense(inputs=h, units=self.config.LAYER_UNITS, use_bias=True,
                                 trainable=True, name="fc3")

            fc = tf.contrib.layers.dropout(fc, self.keep_prob)  # 根据比例keep_prob输出输入数据，最终返回一个张量
            fc = tf.nn.relu(fc)  # 激活函数，此时fc的维度是hidden_dim

            # 分类器,以fc作为输入，输出大小为2
            self.logits = tf.layers.dense(fc, self.config.NUM_CLASS,
                                          name='fc4')  # 将fc从[batch_size,hidden_dim]映射到[batch_size,num_class]输出
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


def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype('float32')


# 返回size
# glorot正常初始化,它从以0为中心的截断正态分布中抽取样本,stddev = sqrt(2 / (fan_in + fan_out)),其中 fan_in 是权重张量的输入单元数,而 fan_out 是权重张量中的输出单位数
def glorot_uniform(size):
    fan_in, fan_out = size
    s = np.sqrt(6. / (fan_in + fan_out))
    return np.random.uniform(size=size, low=-s, high=s).astype("float32")


class BiGRU(object):
    def __init__(self, n_in, n_hidden, n_out, batch_size=128):
        # 建立正向反向两个GRU
        self.gru_1 = GRU(n_in, n_hidden, n_out, batch_size=batch_size)
        self.gru_2 = GRU(n_in, n_hidden, n_out, batch_size=batch_size)

        self.params = self.gru_1.params
        self.params += self.gru_2.params

    def __call__(self, input, return_list=False):
        # 倒置第二维
        # input:[128,30,128]
        # reverse_input:[128,30,128]词序反
        reverse_input = input[:, ::-1, :]

        # 返回两个隐藏向量在第三维的连接，[x,y,2*z]
        res1 = self.gru_1(input, return_list)
        if return_list:
            res2 = self.gru_2(reverse_input, return_list)[:, ::-1, :]
            return tf.concat([res1, res2], axis=2)
        else:
            res2 = self.gru_2(reverse_input, return_list)
        return tf.concat([res1, res2], axis=1)


# 门控递归单元
# 将文本编码为隐藏向量，z,r表示两个方向
class GRU(object):
    def __init__(self, n_in, n_hidden, n_out, activation=tf.nn.tanh, inner_activation=tf.nn.sigmoid,
                 output_type='real', batch_size=128):

        self.activation = activation
        self.inner_activation = inner_activation
        self.output_type = output_type

        self.batch_size = batch_size
        self.n_hidden = n_hidden

        # 递归权重是共享的变量，theano.shared将变量设为全局，borrow=True对数据的改变会影响原始数据
        # U_z[n_hidden,n_hidden]正态
        self.U_z = tf.Variable(ortho_weight(n_hidden), dtype="float32", trainable=True, name='U_z')
        # W_z[n_in,n_hidden]
        self.W_z = tf.Variable(glorot_uniform((n_in, n_hidden)), dtype="float32", trainable=True, name='W_z')
        # b_z[n_hidden,]
        self.b_z = tf.Variable(np.zeros(n_hidden,), dtype="float32", trainable=True, name='b_z')
        # U_r[n_hidden,n_hidden]正态
        self.U_r = tf.Variable(ortho_weight(n_hidden), dtype="float32", trainable=True, name='U_r')
        # W_r[n_in,n_hidden]
        self.W_r = tf.Variable(glorot_uniform((n_in, n_hidden)), dtype="float32", trainable=True, name='W_r')
        # b_r[n_hidden,]
        self.b_r = tf.Variable(np.zeros(n_hidden,), dtype="float32", trainable=True, name='b_r')

        # U_h[n_hidden,n_hidden]正态
        self.U_h = tf.Variable(ortho_weight(n_hidden), dtype="float32", trainable=True, name='U_h')
        # W_h[n_in,n_hidden]
        self.W_h = tf.Variable(glorot_uniform((n_in, n_hidden)), dtype="float32", trainable=True, name='W_h')
        # b_h[n_hidden,]
        self.b_h = tf.Variable(np.zeros(n_hidden,), dtype="float32", trainable=True, name='b_h')

        self.params = [self.W_z, self.W_h, self.W_r,
                       self.U_h, self.U_r, self.U_z,
                       self.b_h, self.b_r, self.b_z]

    def __call__(self, input, return_list=False):
        # 构建循环图(每一步执行的函数，迭代的输入，初始化输出)
        # dimshuffle(1,0,2)表示A.B.C-》B.A.C
        # scan沿着第一维展开输入，所以变序之后，每一步的输入就变成了批次内所有向量的第一个词的向量
        # input[128,30,128],变换后[30,128,128]
        input_1 = tf.transpose(input, perm=[1, 0, 2])
        self.h_l= tf.scan(self.step2, elems=input_1)
        # B.A.C->A.B.C
        self.h_l = tf.transpose(self.h_l, perm=[1, 0, 2])
        if return_list:
            return self.h_l
        # 取第二维的最后一个
        return self.h_l[:, -1, :]

    # 以下两个方法用于产生Sx和Sy的隐藏向量
    # input[128,128]
    def step2(self, x_t, h_tm1):
        # x_z = x_t*W_z+b_z
        x_z = tf.matmul(x_t, self.W_z) + self.b_z
        # x_r = x_t*W_r+b_r
        x_r = tf.matmul(x_t, self.W_r) + self.b_r
        # x_h = x_t*W_h+b_h
        x_h = tf.matmul(x_t, self.W_h) + self.b_h
        # z = sigmoid(x_z+h_tm1*U_z)
        z = self.inner_activation(x_z + tf.matmul(h_tm1, self.U_z))
        # r = sigmoid(x_r+h_tm1*U_r)
        r = self.inner_activation(x_r + tf.matmul(h_tm1, self.U_r))
        # hh = tanh(x_h+r*h_tm1*U_h)
        hh = self.activation(x_h + tf.matmul(r * h_tm1, self.U_h))
        # h = z*h_tm1+(1-z)*hh
        h = z * h_tm1 + (1 - z) * hh
        return h
