import numpy as np
import os
import time
from multiprocessing.dummy import Pool as ThreadPool


def step1(array):
    ######
    # step1 是把数组array分成稀疏和稠密两个数组A，B
    # A 是稀疏
    # B 是稠密
    ######

    array.sort()
    array = array[::-1]

    ################################
    # 本来下面注释部分才是论文的实现，但是按论文的方法的话，集合B被划分出来的区域总是很大
    # 这就导致大部分时间都在做动态规划，而程序执行回溯的时间很短，这就很难体现出并行的优越性了
    # 所以我把代码做了一下调整
    ################################
    A_len = 0
    for i in range(len(array)):
        if array[i] > 500:
            A_len += 1
    #########################

    ######################################
    # tmp_best = np.inf
    # n = len(array)
    # A_len = 0
    # for i in range(n + 1):
    #     left = 2 ** i
    #     right = sum(array[i:])
    #     tmp = max(left, right)
    #     if tmp_best > tmp:
    #         tmp_best = tmp
    #         A_len = i
    # array = np.array(array)
    ######################################

    A = array[:A_len]
    B = array[A_len:]
    return A, B


def step2(B):
    ##########################
    # 动态规划求解B
    ##########################
    if len(B) == 0:
        print('B len is 0!')
        return
    S = sum(B)
    B_len = len(B)
    # 分配数组
    part = np.array([False] * (S + 1))
    part[0] = True
    tmp = part.copy()

    for i in range(1, S + 1):
        for j in range(1, B_len + 1):
            part[i] = tmp[i]
            if i >= B[j - 1]:
                part[i] = part[i] | tmp[i - B[j - 1]]
            tmp[:] = part[:]
    return part


def step3(A, S, Z):
    ############################
    # 并行穷举
    ############################
    n = len(A)

    pool = ThreadPool()

    def paralle(i):
        ans_tmp = format(i, 'b')
        if len(ans_tmp) < n:
            ans_tmp = '0' * (n - len(ans_tmp)) + ans_tmp
        A1 = np.array(list(map(int, ans_tmp)))

        A2 = A1 ^ 1

        A1_sum = A @ A1.T
        A2_sum = A @ A2.T
        T = abs(A1_sum - A2_sum)

        if T > S:
            pass
        else:
            judge1 = S - T
            judge2 = Z[(S - T) // 2]
            if judge1 % 2 == 0 and judge2 == 1:
                # 解
                raise Exception('True')
                # return True
        return False
    try:
        results = pool.map(paralle, range(2**(n-1)))
    except:
        print('==')


if __name__ == '__main__':
    # 测试数据的目录
    data_folder = './testdata'

    # 取出一个测试用
    li = os.listdir(data_folder)
    file_name = 'easy_in_60_.csv'
    file_data = np.loadtxt(os.path.join(data_folder, file_name), dtype=np.int64, delimiter=',')
    print('test file: {}'.format(file_name))

    instance = file_data
    # instance = np.array([ 771, 121, 281, 854, 885, 734, 486, 1003, 83, 62])
    print('test data: {}'.format(instance))
    array = instance.copy()

    # 计时
    start_t = time.time()

    # step1  算法第一步
    A, B = step1(array)
    # A 和 B 是划分出来的结果。A是稀疏集， B是稠密集
    print('A: {}'.format(A))
    print('B: {}'.format(B))

    # 记录集合B的和
    S = sum(B)

    # step 2  步骤二是对稠密集引用动态规划
    Z = step2(B)

    if len(A) == 0:
        # 退化成动态规划算法
        if S % 2 != 0:
            print('S % 2 != 0')
            print(False)
        else:
            print(Z[S // 2])
    else:
        # 步骤三：对集合A回溯
        print('call step3')
        print(step3(A, S, Z))

    end_t = time.time()

    print('time: {}'.format(end_t - start_t))


