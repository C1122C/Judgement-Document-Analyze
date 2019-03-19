import lxml.etree


def get_qw(path):
    tree = lxml.etree.parse(path)
    root = tree.getroot()
    for qw in root:
        return qw


def get_rdss(path):
    content = ''
    qw = get_qw(path)
    for qw_child in qw:
        if qw_child.tag == 'AJJBQK':
            for ajjbqk_child in qw_child:
                if ajjbqk_child.tag == 'BSSLD':
                    for bssld_child in ajjbqk_child:
                        if bssld_child.tag == 'ZJXX':
                            for zjxx_child in bssld_child:
                                if zjxx_child.tag == 'ZJFZ':
                                    for zjfz_child in zjxx_child:
                                        if zjfz_child.tag == 'RDSS':
                                            content = zjfz_child.attrib['value']
    return content


# 指控段落
def get_zkdl(path):
    content = ''
    qw = get_qw(path)
    for qw_child in qw:
        if qw_child.tag == 'AJJBQK':
            for ajjbqk_child in qw_child:
                if ajjbqk_child.tag == 'ZKDL':
                    content = ajjbqk_child.attrib['value']
    return content


# 从新填充了法条内容的文书里提取法条列表
def get_ft_list(path):
    ft_name_list = []
    ft_nr_list = []
    qw = get_qw(path)
    for qw_child in qw:
        if qw_child.tag == 'YYFLNR':
            for yyflfz_child in qw_child:
                if yyflfz_child.tag == 'FLNRFZ':
                    for flnrfz_child in yyflfz_child:
                        flag = 0
                        if flnrfz_child.tag == 'FLMC':
                            fl_mc = flnrfz_child.attrib['value']
                            flag += 1
                        if flnrfz_child.tag == 'FLNR':
                            fl_nr = flnrfz_child.attrib['value']
                            flag += 2
                        if flag == 2 and fl_mc and fl_nr and fl_nr != 'NOT FOUND':
                            ft_name_list.append(fl_mc)
                            ft_nr_list.append(fl_nr)

    return ft_name_list, ft_nr_list


# 文书QW下面的节点内容获取,如文首、诉讼情况、案件基本情况、裁判分析过程、判决结果这几个的value
def get_qw_child_content(path,child_name):
    content = ''
    qw = get_qw(path)
    for qw_child in qw:
        if qw_child.tag == child_name:
            content += qw_child.attrib['value']
    return content


def get_ft_from_qw(path):
    ft_ls = []
    qw = get_qw(path)
    for qw_child in qw:
        if qw_child.tag == 'CPFXGC':
            for cpfxgc_child in qw_child:
                if cpfxgc_child.tag == 'CUS_FLFT_FZ_RY':
                    for fz in cpfxgc_child:
                        if fz.tag == 'CUS_FLFT_RY':
                            ft_ls.append(fz.attrib['value'])
    return ft_ls


# 获取事实内容
def get_ss_match_object(ws_path):
    return get_rdss(ws_path) + get_zkdl(ws_path)


# 获取结论内容
def get_jl_match_object(ws_path):
    return get_qw_child_content(ws_path, 'CPFXGC') + get_qw_child_content(ws_path, 'PJJG')


# 获取交通肇事罪的证据记录列表
def get_zj(ws_path):
    zj_list = []
    qw = get_qw(ws_path)
    for qw_child in qw:
        if qw_child.tag == 'AJJBQK':
            for ajjbqk_child in qw_child:
                if ajjbqk_child.tag == 'BSSLD':
                    for bssld_chid in ajjbqk_child:
                        if bssld_chid.tag == 'ZJXX':
                            for zjxx_child in bssld_chid:
                                if zjxx_child.tag == 'ZJFZ':
                                    for zjfz_child in zjxx_child:
                                        if zjfz_child.tag == 'ZJJL':
                                            zj_list.append(zjfz_child.attrib['value'])
    return zj_list


# 获取xml任意路径的value值
def get_node_content(ws_path,xml_path):
    path_list = xml_path.split('/')
    print(path_list)
    tree = lxml.etree.parse(ws_path)
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
