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
    qw = getQW(path)
    for qwchild in qw:
        # 引用法条内容
        if qwchild.tag == 'YYFLNR':
            for yyflfzchild in qwchild:
                # 法条内容分组
                if yyflfzchild.tag == 'FLNRFZ':
                    for flnrfzchild in yyflfzchild:
                        flag = 0
                        # 法条名称
                        if flnrfzchild.tag == 'FLMC':
                            flmc = flnrfzchild.attrib['value']
                            flag += 1
                        # 法条内容
                        if flnrfzchild.tag == 'FLNR':
                            flnr = flnrfzchild.attrib['value']
                            flag += 2
                        if flag == 2 and flmc and flnr and flnr != 'NOT FOUND':
                            name_list.append(flmc)
                            nr_list.append(flnr)

    return name_list, nr_list


# 提取先验知识
def get_ks(path):
    res = []
    name = ''
    chac = ''
    sec = ''
    qw = getQW(path)
    for qwchild in qw:
        # 先验知识
        if qwchild.tag == 'KS':
            for kschild in qwchild:
                # 名称
                if kschild.tag == 'K_N':
                    name = kschild.attrib['value']
                # 章
                if kschild.tag == 'K_C':
                    chac = kschild.attrib['value']
                # 节
                if kschild.tag == 'K_S':
                    sec = kschild.attrib['value']
    if name != '':
        res.append(name)
    if chac != '':
        res.append(chac)
    if sec != '':
        res.append(sec)
    return res


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
    return content


# 分词
def cut(fact, statue, ks):
    sta = ''
    for s in statue:
        sta = sta + s + ";"
    # 获取停用字典
    stop_list = get_lines('../../source/stopwords.txt')
    cx_save = ['n', 'v', 'a', 'x']

    cut_fact = get_str_segment(fact, cx_save, stop_list)
    cut_statue = get_str_segment(sta, cx_save, stop_list)
    cut_ks = 'ft:' + ' '.join(get_str_segment(ks[0], cx_save, stop_list)) + '@'
    if len(ks) >= 2:
        cut_ks += "章" + ':' + ' '.join(get_str_segment(ks[1], cx_save, stop_list)) + '@'
    if len(ks) >= 3:
        cut_ks += "节" + ':' + ' '.join(get_str_segment(ks[2], cx_save, stop_list)) + '@'
    s = ' '.join(cut_fact) + '|' + ' '.join(cut_statue) + '|' + cut_ks
    return s


# 向量化
def vec(data):
    model_fact = load_models('../../source/wordvector/ssmodel_size128.model')
    model_statue = load_models('../../source/wordvector/lawmodel_size128.model')

    news = ''
    array = data.split('|')
    if len(array) != 3:
        print(data)
        return

    fact_ls = array[0].split(' ')
    statue_ls = array[1].split(' ')
    ks_ls = array[2].split('@')
    for ss in fact_ls:
        news += '//'.join(map(str, list(vector(ss, model_fact)))) + ' '
    news += '|'
    for zw in statue_ls:
        news += '//'.join(map(str, list(vector(zw, model_statue)))) + ' '
    news += '|' + array[3] + '|'
    news += ks_vec(ks_ls, model_statue, flag=3)
    return news


def ks_vec(ks_ls, word_m, flag=3):
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
