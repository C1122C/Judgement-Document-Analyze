import lxml.etree
import numpy as np
import jieba.posseg as pos
from jdac.code.pre_op.word2vec import vector, load_models


# 获取全文
def getQW(path):
    tree = lxml.etree.parse(path)
    root = tree.getroot()
    for qw in root:
        return qw


# 提取法条列表
def get_law_list(path):
    name_list = []
    nr_list = []
    ks_list = []
    qw = getQW(path)
    for qwchild in qw:
        # 引用法条内容
        if qwchild.tag == 'YYFLNR':
            for yyflfzchild in qwchild:
                # 法条内容分组
                if yyflfzchild.tag == 'FLNRFZ':
                    for flnrfzchild in yyflfzchild:
                        ks = []
                        flmc = ''
                        flnr = ''
                        # 法条名称
                        # print(flnrfzchild.tag)

                        if flnrfzchild.tag == 'FLMC':
                            flmc = flnrfzchild.attrib['value']
                            # print("IN FLMC")
                            # print(flnrfzchild.attrib['value'])
                        # 法条内容
                        if flnrfzchild.tag == 'FLNR':
                            flnr = flnrfzchild.attrib['value']
                            # print("IN FLNR")
                            # print(flnrfzchild.attrib['value'])
                        if flnrfzchild.tag == 'KS':
                            for sub in flnrfzchild:
                                # print("IN KS")
                                if sub.tag == 'K_N':
                                    name = sub.attrib['value']
                                    # print(sub.attrib['value'])
                                    ks.append(name)
                                if sub.tag == 'K_C':
                                    c = sub.attrib['value']
                                    # print(sub.attrib['value'])
                                    ks.append(c)
                                if sub.tag == 'K_S':
                                    s = sub.attrib['value']
                                    # print(sub.attrib['value'])
                                    ks.append(s)
                                    ks_list.append(ks)

                        if flmc != '':
                            name_list.append(flmc)
                        if flnr != '':
                            nr_list.append(flnr)

    # print(len(name_list))
    # print(len(nr_list))
    return name_list, nr_list, ks_list


# 获取事实内容
def get_fact(path):
    content = ''
    qw = getQW(path)
    for qwchild in qw:
        # 案件基本情况
        if qwchild.tag == 'AJJBQK':
            for ajjbqkchild in qwchild:
                # 指控段落
                if ajjbqkchild.tag == 'ZKDL':
                    for zkdlchild in ajjbqkchild:
                        if zkdlchild.tag == 'ZKSS':
                            content = zkdlchild.attrib['value']
    # print(content)
    return content


# 分词
def cut(fact, statue, ks):
    # 获取停用字典
    stop_list = get_lines('../../source/stopwords.txt')
    cx_save = ['n', 'v', 'a', 'x']
    res = []

    cut_fact, cut_statue, cut_ks = [], [], []
    for f in fact:
        f_c = get_str_segment(f, cx_save, stop_list)
        cut_fact.append(f_c)
    for s in statue:
        s_c = get_str_segment(s, cx_save, stop_list)
        cut_statue.append(s_c)
    for k in ks:
        # print(k)
        k_c = 'ft:' + ' '.join(get_str_segment(k[0], cx_save, stop_list)) + '@'
        if len(k) >= 2:
            k_c += "章" + ':' + ' '.join(get_str_segment(k[1], cx_save, stop_list)) + '@'
        if len(k) >= 3:
            k_c += "节" + ':' + ' '.join(get_str_segment(k[2], cx_save, stop_list)) + '@'
        cut_ks.append(k_c)

    for i in range(len(cut_fact)):
        for j in range(len(cut_statue)):
            s = ' '.join(cut_fact[i]) + '|' + ' '.join(cut_statue[j]) + '|' + cut_ks[j]
            res.append(s)

    return res


# 向量化
def vec(data):
    model = load_models('../../source/vec_model_size128.model')

    res = []
    for i in range(len(data)):
        news = ""
        array = data[i].split('|')
        if len(array) != 3:
            print(data)
            return

        fact_ls = array[0].split(' ')
        statue_ls = array[1].split(' ')
        ks_ls = array[2].split('@')
        for ss in fact_ls:
            news += '//'.join(map(str, list(vector(ss, model)))) + ' '
        news += '|'
        for zw in statue_ls:
            news += '//'.join(map(str, list(vector(zw, model)))) + ' '
        news += '|'
        news += ks_vec(ks_ls, model)
        # print(news)
        res.append(news)

    return res


def ks_vec(ks_ls, word_m):
    s = ''
    # 将层级目录分为三级：名字/章/节(包含?:),向量化
    matrix = np.zeros(shape=[3, 128], dtype=float)
    for zw in ks_ls:
        if zw.strip() == '':
            continue
        level = zw.split(':')[0].strip()
        content = zw.split(':')[1].strip()
        if level == '':
            continue
        else:
            conls = content.split(' ')
            vectors = []
            for ob in conls:
                vectors.append(list(vector(ob, word_m)))
            vectors = np.mean(np.array(vectors), axis=0)
            if level == 'ft':
                matrix[0] = vectors
            elif level == '章':
                matrix[1] = vectors
            elif level == '节' or level == '?':
                matrix[2] = vectors

    # 将得到的matrix存储到s中
    for line in matrix:
        # print(line)
        s += '//'.join(map(str, list(line))) + ' '
    return s


# 读文件
def get_lines(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read().split('\n')
    lines = list(filter(lambda x: str(x).strip() != '', content))
    return lines


def get_str_segment(content, cx_save, stop_list):
    words = pos.cut(content)
    sls = []
    for word, cx in words:
        if cx in cx_save:
            if word in stop_list:
                pass
            else:
                sls.append(word)
    return sls
