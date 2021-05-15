import struct

def code2txt(code_table, R_n):
    # 编码表(s,o,n,t)
    code_s,code_o,code_n,code_t = code_table

    # 字符流文件
    # 打开文件
    filename = "code.txt"
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

    # 二进制流文件
    # 打开文件
    filename = "code_bin.txt"
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
