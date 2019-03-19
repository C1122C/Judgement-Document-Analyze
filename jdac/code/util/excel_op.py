import xlwt
import xlrd
from xlutils.copy import copy


# rows:list
# columns:list
# data:list[list]
def create_x(ws_name, rows, columns, data, dir):
    fnt = xlwt.Font()
    fnt.name = 'SimSun'
    fnt.bold = True
    fnt.height = 250
    wb = xlwt.Workbook()
    style = xlwt.easyxf('align: wrap on;')
    style.font = fnt
    ws = wb.add_sheet(ws_name)
    # 设置行
    for i in range(len(rows)):
        ws.write(i + 1, 0, rows[i], style)

    # 设置列
    for i in range(len(columns)):
        ws.write(0, i + 1, columns[i], style)

    for i in range(len(columns)+1):
        ws.col(i).width = 256 * 40

    ws.panes_frozen = True
    ws.horz_split_pos = 1

    # 录入数据
    for i in range(len(data)):
        for j in range(len(data[i])):
            ws.write(i+1, j+1, data[i][j])
    wb.save(dir+'/'+ws_name+'_20181110.xls')


def get_cols(excel_path):
    excel_file = xlrd.open_workbook(excel_path)
    sheet = excel_file.sheet_by_index(0)
    row_ls = []
    for i in range(1, sheet.ncols):
        cell = sheet.cell_value(0, i)
        row_ls.append(cell)
    return row_ls


def get_rows2(excel_path):
    excel_file = xlrd.open_workbook(excel_path)
    sheet = excel_file.sheet_by_index(0)
    cols = []
    for i in range(1, sheet.nrows):
        cell = sheet.cell_value(i, 1)
        cols.append(cell)
    return cols


def get_rows(excel_path):
    excel_file = xlrd.open_workbook(excel_path)
    sheet = excel_file.sheet_by_index(0)
    cols = []
    for i in range(1, sheet.nrows):
        cell = sheet.cell_value(i,0)
        cols.append(cell)
    return cols


def get_excel_data(excel_path):
    data = []
    excel_file = xlrd.open_workbook(excel_path)
    sheet = excel_file.sheet_by_index(0)
    for i in range(1, sheet.ncols):
        col = []
        for j in range(1, sheet.nrows):
            cell = sheet.cell_value(j, i)
            if isinstance(cell, float):
                col.append(cell)
            if isinstance(cell, str):
                if cell.strip() == '':
                    col.append(0)
                else:
                    col.append(int(cell[0]))
        data.append(col)
    return data


def alter_excel(excel_path, cols, rows, datas):
    p = xlrd.open_workbook(excel_path)
    wb = copy(p)
    sheet = wb.get_sheet(0)
    print(sheet.name)
    p_data = zip(cols, rows, datas)
    for col, row, data in p_data:
        sheet.write(int(row.strip())+1, int(col.strip())+1, data)
    wb.save(excel_path)
