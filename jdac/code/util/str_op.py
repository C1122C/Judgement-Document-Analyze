import re
import jieba.posseg as pos


def cut_content(content):
    return list(filter(lambda x: x.strip() != '', re.split('；|。', content)))


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


def show_class_num(content):
    lines = content.split('\n')
    class_0, class_1 = 0, 0

    for line in lines:
        if len(line.split('|')) < 5:
            continue
        label = line.split('|')[3][0]
        if label == '0':
            class_0 += 1
        else:
            class_1 += 1
    print('0:', class_0)
    print('1:', class_1)
