import struct
import numpy as np

# 写入字符流文件
def code2txt_str(code_table, R_n, filename):
    # 编码表(s,o,n,t)
    code_s,code_o,code_n,code_t = code_table
    # 打开文件（字符流）
    file = open(filename, "w")
    # 写入文件
    file.write('########### code_s ###########\n')
    for r in range(R_n):
        file.write("{} {}\n".format(r, code_s[r]))
    file.write('########### code_o ###########\n')
    for r in range(R_n):
        file.write("{} {}\n".format(r, code_o[r]))
    file.write('########### code_n ###########\n')
    for r in range(R_n):
        file.write("{} {}\n".format(r, code_n[r]))
    file.write('########### code_t ###########\n')
    for r in range(R_n):
        file.write("{} {}\n".format(r, code_t[r]))
    # 关闭文件
    file.close()

# 写入二进制文件
def code2txt_bin(code_table, R_n, filename):
    # 编码表(s,o,n,t)
    code_s,code_o,code_n,code_t = code_table
    # 打开文件（二进制）
    file = open(filename, "wb")
    # 写入文件
    #file.write('########### code_s ###########\n')
    for r in range(R_n):
        bin_code = struct.pack('f',code_s[r])
        file.write(bin_code)
    #file.write('########### code_o ###########\n')
    for r in range(R_n):
        bin_code = struct.pack('f',code_o[r])
        file.write(bin_code)
    #file.write('########### code_n ###########\n')
    for r in range(R_n):
        bin_code = struct.pack('h',code_n[r])
        file.write(bin_code)
    #file.write('########### code_t ###########\n')
    for r in range(R_n):
        bin_code = struct.pack('h',code_t[r])
        file.write(bin_code)   
    # 关闭文件
    file.close()   

# 写入二进制文件（量化）
def code2txt_qtz_bin(code_table, R_n, filename, s_bit, o_bit):
    # 编码表(s,o,n,t)
    code_s,code_o,code_n,code_t = code_table
    # 打开文件（二进制）
    file = open(filename, "wb")
    # 写入文件
    #file.write('########### code_s ###########\n')
    for r in range(R_n):
        code = int((code_s[r] * (2 << s_bit)) + 0.5)
        bin_code = struct.pack('c',code)
        file.write(bin_code)
    #file.write('########### code_o ###########\n')
    for r in range(R_n):
        code = int((code_o[r] * (2 << o_bit)) + 0.5)
        bin_code = struct.pack('c',code)
        file.write(bin_code)
    #file.write('########### code_n ###########\n')
    for r in range(R_n):
        bin_code = struct.pack('h',code_n[r])
        file.write(bin_code)
    #file.write('########### code_t ###########\n')
    for r in range(R_n):
        bin_code = struct.pack('c',code_t[r])
        file.write(bin_code)   
    # 关闭文件
    file.close() 

# 读取字符流文件
def txt2code_str(R_n, filename):
    # 编码表(s,o,n,t)
    code_s = np.empty(R_n)  # 放缩因子
    code_o = np.empty(R_n)  # 偏移量
    code_n = np.empty(R_n, np.int)  # 父图编号
    code_t = np.empty(R_n, np.int)  # 变换方式

    # 打开文件（字符流）
    file = open(filename, "r")
    # 读取文件
    ########### code_s ###########
    file.readline()
    for r in range(R_n):
        i, code_s[r] = file.readline().split()
    ########### code_o ###########
    file.readline()
    for r in range(R_n):
        i, code_o[r] = file.readline().split()
    ########### code_n ###########
    file.readline()
    for r in range(R_n):
        i, code_n[r] = file.readline().split()
    ########### code_t ###########
    file.readline()
    for r in range(R_n):
        i, code_t[r] = file.readline().split()
    # 关闭文件
    file.close()
    # 返回编码表(s,o,n,t)
    return [code_s, code_o, code_n, code_t]

# 读取二进制文件
def txt2code_bin(R_n, filename):
    # 编码表(s,o,n,t)
    code_s = np.empty(R_n)  # 放缩因子
    code_o = np.empty(R_n)  # 偏移量
    code_n = np.empty(R_n, np.int)  # 父图编号
    code_t = np.empty(R_n, np.int)  # 变换方式

    # 打开文件（二进制）
    file = open(filename, "rb")
    # 读取文件
    ########### code_s ###########
    for r in range(R_n):
        code_s[r] = struct.unpack('f',file.read(4))[0]
    ########### code_o ###########
    for r in range(R_n):
        code_o[r] = struct.unpack('f',file.read(4))[0]
    ########### code_n ###########
    for r in range(R_n):
        code_n[r] = struct.unpack('h',file.read(2))[0]
    ########### code_t ###########
    for r in range(R_n):
        code_t[r] = struct.unpack('h',file.read(2))[0]
    # 关闭文件
    file.close()
    # 返回编码表(s,o,n,t)
    return [code_s, code_o, code_n, code_t]

# 读取二进制文件（量化）
def txt2code_qtz_bin(R_n, filename, s_bit, o_bit):
    # 编码表(s,o,n,t)
    code_s = np.empty(R_n)  # 放缩因子
    code_o = np.empty(R_n)  # 偏移量
    code_n = np.empty(R_n, np.int)  # 父图编号
    code_t = np.empty(R_n, np.int)  # 变换方式

    # 打开文件（二进制）
    file = open(filename, "rb")
    # 读取文件
    ########### code_s ###########
    for r in range(R_n):
        code = struct.unpack('h',file.read(1))[0]
        code_s[r] = code/(2 << s_bit)
    ########### code_o ###########
    for r in range(R_n):
        code = struct.unpack('h',file.read(1))[0]
        code_o[r] = code/(2 << o_bit)
    ########### code_n ###########
    for r in range(R_n):
        code_n[r] = struct.unpack('h',file.read(2))[0]
    ########### code_t ###########
    for r in range(R_n):
        code_t[r] = struct.unpack('h',file.read(1))[0]
    # 关闭文件
    file.close()
    # 返回编码表(s,o,n,t)
    return [code_s, code_o, code_n, code_t]