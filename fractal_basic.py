import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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
    Rp = np.empty([R_n, R_len, R_len])       # R池
    for i in range(R_row):
        for j in range(R_col):
            # 将原图分为R_len*R_len大小的子图，互不重叠
            Rp[i*R_col+j] = I_in[i*R_len:(i+1)*R_len, j*R_len:(j+1)*R_len]

    # 存储父图及其变换
    T_n = 8     # 变换数
    Dp = np.empty([D_n, T_n, R_len, R_len])  # D池
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
    R_seq = np.empty([R_n, R_len*R_len])
    for i in range(R_n):
        R_seq[i] = Rp[i].reshape(1,-1)[0]
    # 父图序列化
    D_seq = np.empty([D_n, T_n, R_len*R_len])
    for i in range(D_n):
        for j in range(T_n):
            D_seq[i][j] = Dp[i][j].reshape(1,-1)[0]

    # 找R的最佳匹配D，记录SE(R,D)最小时的分形编码
    code_s = np.empty(R_n)  # 放缩因子
    code_o = np.empty(R_n)  # 偏移量
    code_n = np.empty(R_n)  # 父图编号
    code_t = np.empty(R_n)  # 变换方式
    for i in range(R_n):
        print("i =", i)
        R_mean = R_seq[i].mean()
        R_diff = R_seq[i] - R_mean
        mse = -1    # 最小平方误差
        for j in range(D_n):
            for k in range(T_n):
                D_mean = D_seq[j][k].mean()
                D_diff = D_seq[j][k] - D_mean
                # R = s*D + o
                sum_D_diff = sum(D_diff)
                scale = 0
                if sum_D_diff != 0:
                    scale = np.dot(R_diff,D_diff)/(sum_D_diff**2)      # 放缩因子
                ofs = R_mean - scale*D_mean     # 偏移量
                se = np.sum(scale*D_seq[j][k] + ofs - R_seq[i])**2  # 平方误差
                # 存储最小平方误差对应的分形编码
                if mse < 0 or se < mse:
                    mse = se
                    code_s[i] = scale   # 放缩因子
                    code_o[i] = ofs     # 偏移量
                    code_n[i] = j       # 父图编号
                    code_t[i] = k       # 变换方式
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
        print("------第",k,"次迭代------")
        I_temp = I_out.copy()
        for r in range(R_n):
            x = int(code_n[r]//D_col)
            y = int(code_n[r]%D_col)
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
    img_path = "demo.jpg"
    I_src = np.asarray(Image.open(img_path).convert('L'))   # 转灰度图
    # np.asarray
    plt.imshow(I_src)   # 原图
    plt.show()

    # 子图/父图大小
    B_len = 8       # 块长
    R_len = B_len   # 子图大小B*B
    D_len = 2*B_len # 父图大小2B*2B
    D_ofs = B_len   # 父图步长B
    # 分形编码
    code_table = fractal_encode(I_src, R_len, D_len, D_ofs)
    # 分形解码
    iter_n = 1
    I_init = np.zeros(I_src.shape, np.uint8)
    plt.imshow(I_init)   # 初始图
    plt.show()

    I_out = fractal_decode(code_table, I_init, iter_n, R_len, D_len, D_ofs)
    plt.imshow(I_out)   # 恢复图
    plt.show()
    