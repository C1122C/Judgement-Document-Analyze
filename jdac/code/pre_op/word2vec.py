# 输入法条的关键词，使用word2vec获取法条关键词的同义词，注意:输入文书作为语料库，用文书的QW内容
import os
import jieba.posseg as pos
import gensim
from gensim.models import word2vec


def build_model(doc_path, corpus_path, model_path, stop_word_path):
    print('build model......')
    set_cor(doc_path, corpus_path, stop_word_path)
    print('start.....')
    sentence = word2vec.LineSentence(corpus_path)
    model = word2vec.Word2Vec(sentence, min_count=5, size=128)
    print('saving.....')
    model.save(model_path)
    print('built......')
    print('end....')


# 将全文中指定词性的词删除
def filter_word_with_cx(cutre, cxlist, spwordpath):
    word_list = []
    print('filter word....')
    stopwords = get_lines(spwordpath)
    for (w, k) in cutre:
        if k not in cxlist and w not in stopwords:
            word_list.append(w)
    return word_list


def set_cor(dic_path, corpus_path, spword_path):
    print('setCor:'+dic_path)
    file_path_list = os.listdir(dic_path)
    index = 0
    cx_list = ['x', 'p', 'nr', 'uj']
    with open(corpus_path, 'w', encoding='UTF-8') as f:
        for filepath in file_path_list:
            print('index', index)
            index += 1
            # content = getQW(os.path.join(dic_path,filepath)).attrib['value']
            content = open(os.path.join(dic_path, filepath), 'r', encoding='utf-8').read()
            content_cut = pos.cut(content)
            content_filter = filter_word_with_cx(content_cut, cx_list, spword_path)
            for word in content_filter:
                f.write(word+' ')
            f.write('\n')


def load_models(model_path):
    return gensim.models.Word2Vec.load(model_path)


def vector(v, model):
    try:
        return model[v]
    except:
        return [0]*128


def get_lines(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read().split('\n')
    lines = list(filter(lambda x: str(x).strip() != '', content))
    return lines