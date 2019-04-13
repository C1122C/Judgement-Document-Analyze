import lxml.etree


def getQW(path):
    tree = lxml.etree.parse(path)
    root = tree.getroot()
    for qw in root:
        return qw


def getRDSS(path):
    content = ''
    qw = getQW(path)
    for qwchild in qw:
        if qwchild.tag == 'AJJBQK':
            for ajjbqkchild in qwchild:
                if ajjbqkchild.tag == 'BSSLD':
                    for bssldchild in ajjbqkchild:
                        if bssldchild.tag == 'ZJXX':
                            for zjxxchild in bssldchild:
                                if zjxxchild.tag == 'ZJFZ':
                                    for zjfzchild in zjxxchild:
                                        if zjfzchild.tag == 'RDSS':
                                            content = zjfzchild.attrib['value']
    return content


# 指控段落
def getZKDL(path):
    content = ''
    qw = getQW(path)
    for qwchild in qw:
        if qwchild.tag == 'AJJBQK':
            for ajjbqkchild in qwchild:
                if ajjbqkchild.tag == 'ZKDL':
                    content = ajjbqkchild.attrib['value']
    return content


# 从新填充了法条内容的文书里提取法条列表
def get_law_list(path):
    name_list = []
    nr_list = []
    qw = getQW(path)
    for qwchild in qw:
        if qwchild.tag == 'YYFLNR':
            for yyflfzchild in qwchild:
                if yyflfzchild.tag == 'FLNRFZ':
                    for flnrfzchild in yyflfzchild:
                        flag = 0
                        if flnrfzchild.tag == 'FLMC':
                            flmc = flnrfzchild.attrib['value']
                            flag += 1
                        if flnrfzchild.tag == 'FLNR':
                            flnr = flnrfzchild.attrib['value']
                            flag += 2
                        if flag == 2 and flmc and flnr and flnr != 'NOT FOUND':
                            name_list.append(flmc)
                            nr_list.append(flnr)

    return name_list, nr_list


# 文书QW下面的节点内容获取,如文首、诉讼情况、案件基本情况、裁判分析过程、判决结果这几个的value
def getQWChildContent(path, childname):
    content = ''
    qw = getQW(path)
    for qwchild in qw:
        if qwchild.tag == childname:
            content += qwchild.attrib['value']

    return content


def get_law_from_doc(path):
    ftls = []
    qw = getQW(path)
    for qwchild in qw:
        if qwchild.tag == 'CPFXGC':
            for cpfxgcchild in qwchild:
                if cpfxgcchild.tag == 'CUS_FLFT_FZ_RY':
                    for fz in cpfxgcchild:
                        if fz.tag == 'CUS_FLFT_RY':
                            ftls.append(fz.attrib['value'])
    return ftls


# 获取事实内容
def get_fact(doc_path):
    return getRDSS(doc_path) + getZKDL(doc_path)


# 获取xml任意路径的value值
def get_node_content(doc_path, xml_path):
    path_list = xml_path.split('/')
    print(path_list)
    tree = lxml.etree.parse(doc_path)
    root = tree.getroot()
    point = root
    index = 0
    while index < len(path_list):
        for child in point:
            if child.tag == path_list[index]:
                point = child
                index += 1
                break
    value_list = []
    parent = point.getparent()
    for p in parent:
        if p.tag == path_list[-1]:
            value_list.append(p.attrib['value'])
