import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import code_text
from skimage.measure import compare_mse, compare_psnr, compare_ssim

# 矩阵变换参数
class Trans:
    FLIP_LEFT_RIGHT = 1
    FLIP_UP_DOWN    = 2
    FLIP_DIAG_MAIN  = 3
    FLIP_DIAG_SUB   = 4
    ROTATE_90   = 5
    ROTATE_180  = 6
    ROTATE_270  = 7

# 矩阵变换
def mtranspose(mat, mode):
    mat_cp = mat.copy()
    row_n, col_n = mat.shape
    n = min(row_n, col_n)   # 若行列数不相等，取较小
    # 上下翻转
    if mode == Trans.FLIP_LEFT_RIGHT:
        mat_cp = np.fliplr(mat)
    # 左右翻转
    elif mode == Trans.FLIP_UP_DOWN:
        mat_cp = np.flipud(mat)
    # 沿主对角线翻转
    elif mode == Trans.FLIP_DIAG_MAIN:
        for i in range(n):
            for j in range(n):
                mat_cp[j][i] = mat[i][j]
    # 沿副对角线翻转
    elif mode == Trans.FLIP_DIAG_SUB:
        for i in range(n):
            for j in range(n):
                mat_cp[n-j-1][n-i-1] = mat[i][j]
    # 旋转90
    elif mode == Trans.ROTATE_90:
        mat_cp = np.rot90(mat, 1)
    # 旋转180
    elif mode == Trans.ROTATE_180:
        mat_cp = np.rot90(mat, 2)
    # 旋转270
    elif mode == Trans.ROTATE_270:
        mat_cp = np.rot90(mat, 3)
    # 返回变换后的矩阵
    return mat_cp

# 分形编码
def fractal_encode(I_in, R_len, D_len, D_ofs):
    # 图片行列数
    row_n, col_n = I_in.shape
    # 计算子图数量
    R_row = row_n//R_len
    R_col = col_n//R_len
    R_n = R_row * R_col
    print("子图数量：",R_n)
    # 计算父图数量
    D_row = (row_n - D_len)//D_ofs + 1
    D_col = (col_n - D_len)//D_ofs + 1
    D_n = D_row * D_col
    print("父图数量：",D_n)
    # 存储子图（Rn*Rl*Rl）
    Rp = np.empty([R_n, R_len, R_len], np.uint8)       # R池
    for i in range(R_row):
        for j in range(R_col):
            # 将原图分为R_len*R_len大小的子图，互不重叠
            Rp[i*R_col+j] = I_in[i*R_len:(i+1)*R_len, j*R_len:(j+1)*R_len]

    # 存储父图及其变换
    T_n = 8     # 变换数
    Dp = np.empty([D_n, T_n, R_len, R_len], np.uint8)  # D池
    for i in range(D_row):
        for j in range(D_col):
            # 将原图分为D_len*D_len大小的父图，重叠步长D_ofs
            temp = I_in[i*D_ofs:i*D_ofs+D_len, j*D_ofs:j*D_ofs+D_len]
            # 对2*2块取均值，使父图大小从2B*2B变为B*B，与子图相同
            D_temp = np.empty((R_len, R_len),np.uint8)
            for Di in range(R_len):
                for Dj in range(R_len):
                    D_temp[Di][Dj] = temp[2*Di:2*(Di+1),2*Dj:2*(Dj+1)].mean()
            # 父图变换
            k = i*D_col + j
            Dp[k][0] = D_temp    # 原图
            for t in range(1, T_n):
                Dp[k][t] = mtranspose(D_temp, t)

    # 子图序列化
    R_seq = np.empty([R_n, R_len*R_len],np.uint8)
    for i in range(R_n):
        R_seq[i] = Rp[i].reshape(1,-1)[0]
    # 父图序列化
    # 注：对Tn个变换，父图序列的均值、差值序列的模长一样，只需算一次
    D_seq = np.empty([D_n, T_n, R_len*R_len],np.uint8)
    D_mean = np.empty(D_n)        # 均值
    D_diff = np.empty((D_n,T_n, R_len*R_len))   # 差值
    D_diff_norm2 = np.empty(D_n)  # 模**2
    for i in range(D_n):
        j = 0
        # 父图序列的均值、差值序列的模长只需算一次
        D_seq[i][j] = Dp[i][j].reshape(1,-1)[0]
        D_mean[i] = D_seq[i][j].mean()
        D_diff[i][j] = D_seq[i][j] - D_mean[i]
        D_diff_norm2[i] = np.dot(D_diff[i][j],D_diff[i][j])
        # 其他变换中，只需算父图序列、差值序列
        for j in range(1, T_n):
            D_seq[i][j] = Dp[i][j].reshape(1,-1)[0]
            D_diff[i][j] = D_seq[i][j] - D_mean[i]

    # 找R的最佳匹配D，记录SE(R,D)最小时的分形编码
    code_s = np.empty(R_n)  # 放缩因子
    code_o = np.empty(R_n)  # 偏移量
    code_n = np.empty(R_n, np.int)  # 父图编号
    code_t = np.empty(R_n, np.int)  # 变换方式
    print("查找每个子图的最佳匹配......")
    for i in range(R_n):
        if i % 10 == 0:
            print("i =", i)
        R_mean = R_seq[i].mean()
        R_diff = R_seq[i] - R_mean
        mse = -1    # 最小平方误差
        for j in range(D_n):
            for k in range(T_n):
                scale = 0
                if D_diff_norm2[j] != 0:
                    scale = np.dot(R_diff,D_diff[j][k])/D_diff_norm2[j]    # 放缩因子
                ofs = R_mean - scale*D_mean[j]     # 偏移量
                diff_seq = scale*D_seq[j][k] + ofs - R_seq[i]
                se = np.dot(diff_seq, diff_seq)  # 平方误差
                # 存储最小平方误差对应的分形编码
                if mse < 0 or se < mse:
                    mse = se
                    code_s[i] = scale   # 放缩因子
                    code_o[i] = ofs     # 偏移量
                    code_n[i] = j       # 父图编号
                    code_t[i] = k       # 变换方式
        #print("mse =",mse)
    # 返回分形编码(s,o,n,t)
    return [code_s,code_o,code_n,code_t]

# 分形解码
def fractal_decode(code_table, I_init, iter_n, R_len, D_len, D_ofs):
    # 编码表(s,o,n,t)
    code_s,code_o,code_n,code_t = code_table
    # 复制初始图
    row_n, col_n = I_init.shape
    I_out = I_init.copy()
    # 计算子图数量
    R_row = row_n//R_len
    R_col = col_n//R_len
    R_n = R_row * R_col
    # 计算父图列数
    D_col = (col_n - D_len)//D_ofs + 1

    # 恢复图片（迭代）
    for k in range(iter_n):
        print("------第",k+1,"次迭代------")
        I_temp = I_out.copy()
        for r in range(R_n):
            x = code_n[r]//D_col
            y = code_n[r]%D_col
            # 将原图分为D_len*D_len大小的父图，重叠步长D_ofs
            temp = I_out[x*D_ofs:x*D_ofs+D_len, y*D_ofs:y*D_ofs+D_len]
            # 对2*2块取均值，使父图大小从2B*2B变为B*B，与子图相同
            D_temp = np.empty((R_len, R_len),np.uint8)
            for Di in range(R_len):
                for Dj in range(R_len):
                    D_temp[Di][Dj] = temp[2*Di:2*(Di+1), 2*Dj:2*(Dj+1)].mean()
            # 等距变换
            if code_t[r] != 0:      # 恒定变换以外的变换
                D_temp = mtranspose(D_temp, code_t[r])
            # R = s*D + o
            R_temp = code_s[r]*D_temp + code_o[r]
            Ri = r//R_col
            Rj = r%R_col
            I_temp[Ri*R_len:(Ri+1)*R_len, Rj*R_len:(Rj+1)*R_len] = R_temp
        # 迭代图片
        I_out = I_temp.copy()

    # 返回解码图片
    return I_out

if __name__ == '__main__':
    # 读取图片
    img_path = "data/demo256.jpg"
    I_src = np.asarray(Image.open(img_path).convert('L'))   # 转灰度图
    # np.asarray
    plt.imshow(I_src, cmap='gray')   # 原图
    plt.show()

    # 子图/父图大小
    I_row = I_src.shape[0]
    B_len = I_row >> 6   # 块长：通常取4,8,16...
    R_len = B_len   # 子图大小B*B
    D_len = 2*B_len # 父图大小2B*2B
    D_ofs = B_len   # 父图步长B
    ################### 分形编码 ###################
    print("********* 开始编码 *********")
    code_table = fractal_encode(I_src, R_len, D_len, D_ofs)
    print("********* 生成编码文件 *********")
    file_code_str = "code.txt"
    file_code_bin = "code_bin.txt"
    code_text.code2txt_str(code_table, len(code_table[0]), file_code_str)
    code_text.code2txt_bin(code_table, len(code_table[0]), file_code_bin)

    ################### 分形解码 ###################
    # 图片行列数
    row_n, col_n = I_src.shape
    # 计算子图数量
    R_row = row_n//R_len
    R_col = col_n//R_len
    R_n = R_row * R_col
    # 获取编码表
    #file_code_str = "code.txt"
    file_code_bin = "code_bin_4096.txt"
    # code_table = code_text.txt2code_str(R_n, file_code_bin)
    code_table = code_text.txt2code_bin(R_n, file_code_bin)
    # 量化
    s_bit, o_bit = 3, 5
    for i in range(R_n):
        factor = (2<<s_bit)
        code_table[0][i] = int(code_table[0][i]*factor+0.5)/factor
    for i in range(R_n):
        factor = (2<<o_bit)
        code_table[1][i] = int(code_table[1][i]*factor+0.5)/factor

    I_init = np.zeros(I_src.shape, np.uint8)    # 初始码本（全0图片）
    plt.imshow(I_init, cmap='gray')
    plt.show()
    print("********* 开始解码 *********")
    seq_n = 20
    mse_seq = np.empty(seq_n)   # 均方误差
    psnr_seq = np.empty(seq_n)  # 峰值信噪比
    ssim_seq = np.empty(seq_n)  # 结构相似性
    iter_n = 1
    I_out = I_init
    for i in range(seq_n):
        I_out = fractal_decode(code_table, I_out, iter_n, R_len, D_len, D_ofs)
        # 衡量指标
        mse_seq[i] = compare_mse(I_src, I_out)      # 均方误差
        psnr_seq[i] = compare_psnr(I_src, I_out)    # 峰值信噪比
        ssim_seq[i] = compare_ssim(I_src, I_out)    # 结构相似性
        # 显示恢复图
        plt.imshow(I_out, cmap='gray')
        plt.show()

    # 查看衡量指标
    print("均方误差MSE：\n", mse_seq)
    print("峰值信噪比PSNR：\n", psnr_seq)
    print("结构相似性SSIM：\n", ssim_seq)

    # 绘制图表
    if seq_n > 12:
        seq_n = 12
    plt.rcParams["font.family"] = "SimHei"
    plt.rcParams["font.size"] = "14"
    # MSE
    plt.xlabel("迭代次数")
    plt.ylabel("均方误差MSE")
    plt.plot(range(seq_n), mse_seq[:seq_n])
    plt.show()
    # PSNR
    plt.xlabel("迭代次数")
    plt.ylabel("峰值信噪比PSNR")
    plt.plot(range(seq_n), psnr_seq[:seq_n])
    plt.show()
    # SSIM
    plt.xlabel("迭代次数")
    plt.ylabel("结构相似性SSIM")
    plt.plot(range(seq_n), ssim_seq[:seq_n])
    plt.show()

