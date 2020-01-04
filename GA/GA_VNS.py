import numpy as np
import matplotlib.pyplot as plt
import random
import time
import os
from tqdm import tqdm

class GA_partition(object):
    """
    遗传算法解决 partition problem
    """

    def __init__(self, number, iter_number, bag_capacity, data):
        """
        参数初始化
        :param number:
        :param iter_number:
        """
        self.data = data
        self.length = len(self.data)  # 确定染色体编码长度
        self.number = number  # 确定初始化种群数量
        self.iteration = iter_number  # 设置迭代次数
        self.bag_capacity = bag_capacity  # 子集和最多是多少

        self.retain_rate = 0.2  # 每一代精英选择出前20%
        self.random_selection_rate = 0.6  # 对于不是前20%的，有0.5的概率可以进行繁殖
        self.mutation_rate = 0.5  # 变异概率0.01

    def initial_population(self):
        """
        种群初始化，

        :return: 返回种群集合
        """
        init_population = np.random.randint(low=0, high=2, size=[self.length, self.number], dtype=np.int16)
        return init_population

    def subtract_sum(self, chromosome):
        return -(self.bag_capacity - chromosome.T @ self.data)

    def fitness_function(self, chromosome):
        """
        计算适应度函数
        1.  - （最大子集和 - 子集和）
        2.
        :param chromosome:
        :return:
        """

        judge = self.subtract_sum(chromosome)
        if judge > 0:
            value = -self.bag_capacity
        else:
            value = judge

        return value


    def fitness_average(self, init_population):
        """
        求出这个种群的平均适应度，才能知道种群已经进化好了
        :return:返回的是一个种群的平均适应度
        """
        f_accumulation = 0
        for z in range(init_population.shape[1]):
            f_tem = self.fitness_function(init_population[:, z])
            f_accumulation = f_accumulation + f_tem
        f_accumulation = f_accumulation / init_population.shape[1]
        return f_accumulation

    def selection(self, init_population):
        """
        选择
        :param init_population:
        :return: 返回选择后的父代，数量是不定的
        """
        #         sort_population = np.array([[], [], [], [], [], []])  # 生成一个排序后的种群列表，暂时为空
        sort_population = np.array([[]] * (self.length + 1))
        for i in range(init_population.shape[1]):
            x1 = init_population[:, i]
            x2 = self.fitness_function(x1)
            x = np.r_[x1, x2]

            sort_population = np.c_[sort_population, x]

        sort_population = sort_population.T[np.lexsort(sort_population)].T  # 联合排序，从小到大排列

        # 选出适应性强的个体，精英选择
        retain_length = sort_population.shape[1] * self.retain_rate

        #         parents = np.array([[], [], [], [], [], []])  # 生成一个父代列表，暂时为空
        parents = np.array([[]] * (self.length + 1))
        for j in range(int(retain_length)):
            y1 = sort_population[:, -(j + 1)]
            parents = np.c_[parents, y1]

        # print(parents.shape[1])

        rest = sort_population.shape[1] - retain_length  # 精英选择后剩下的个体数
        for q in range(int(rest)):

            if np.random.random() < self.random_selection_rate:
                y2 = sort_population[:, q]
                parents = np.c_[parents, y2]

        parents = np.delete(parents, -1, axis=0)  # 删除最后一行，删除了f值
        # print('打印选择后的个体数')
        # print(parents.shape[0])

        parents = np.array(parents, dtype=np.int16)

        return parents

    def crossover(self, parents):
        """
        交叉生成子代，和初始化的种群数量一致
        :param parents:
        :return:返回子代
        """
        #         children = np.array([[], [], [], [], []])  # 子列表初始化
        children = np.array([[]] * self.length)

        while children.shape[1] < self.number:
            father = np.random.randint(0, parents.shape[1] - 1)
            mother = np.random.randint(0, parents.shape[1] - 1)
            if father != mother:
                # 随机选取交叉点
                cross_point = np.random.randint(0, self.length)
                # 生成掩码，方便位操作
                # mark = 0
                # for i in range(cross_point):
                #     mark |= (1 << i)
                # father = parents[:, father]
                # # print(father)
                # mother = parents[:, mother]
                #
                # # print(mark)
                # # 子代将获得父亲在交叉点前的基因和母亲在交叉点后（包括交叉点）的基因
                # try:
                #     child = ((father & mark) | (mother & ~mark)) & ((1 << self.length) - 1)
                # except:
                #     print('mark:', mark)
                #     print('father:', father)
                #     print('mother:', mother)
                #     raise Exception()

                father = parents[:, father]
                mother = parents[:, mother]

                child = np.zeros_like(father)
                child[:cross_point] = father[:cross_point]
                child[cross_point:] = mother[cross_point:]

                children = np.c_[children, child]

                # 经过繁殖后，子代的数量与原始种群数量相等，在这里可以更新种群。
                # print('子代数量', children.shape[1])
        # print(children.dtype)
        children = np.array(children, dtype=np.int16)
        return children

    def mutation(self, children):
        """
        变异

        :return:
        """
        for i in range(children.shape[1]):

            if np.random.random() < self.mutation_rate:
                j = np.random.randint(0, self.length - 1)  # s随机产生变异位置
                #                 children[:, i] ^= 1 << j  # 产生变异
                children[j, i] = children[j, i] ^ 1
        children = np.array(children, dtype=np.int16)
        return children

    def plot_figure(self, iter_plot, f_plot, f_set_plot):
        """
        画出迭代次数和平均适应度曲线图
        画出迭代次数和每一步迭代最大值图
        :return:
        """
        plt.figure()

        plt.subplot(1, 2, 1)
        plt.plot(iter_plot, f_plot)

        plt.subplot(1, 2, 2)
        plt.plot(iter_plot, f_set_plot)
        plt.show()

    def neighborhood_struct_0(self, instance):
        # instance: 个体，如：[0,1,1,0]
        # 只改变一个Bit,生成其领域：[1,1,1,0],[0,0,1,0],[0,1,0,0],[0,1,1,1]
        assert len(instance) == self.length, '长度不对'
        neighborhoods = np.tile(instance, (self.length,1))
        neighborhoods = neighborhoods.T

        tmp = np.eye(self.length)
        logit = np.logical_xor(neighborhoods, tmp)
        tmp2 = np.zeros_like(neighborhoods, dtype=np.int)
        tmp2[logit] = 1
        neighborhoods = tmp2

        return neighborhoods

    def neighborhood_struct_1(self, instance):
        # 随机生成两个不同的位置
        assert len(instance) == self.length
        neighborhoods = np.tile(instance, (self.length,1))
        neighborhoods = neighborhoods.T

        eyes = np.eye(self.length)
        index = [(i+1)%self.length for i in range(self.length)]
        moves = eyes[index]
        logit = np.logical_xor(neighborhoods, (eyes+moves.T))
        tmp2 = np.zeros_like(neighborhoods, dtype=np.int)
        tmp2[logit] = 1
        neighborhoods = tmp2

        return neighborhoods

    def neighborhood_struct_n(self, instance, n=3):
        assert len(instance) == self.length
        neighborhoods = np.tile(instance, (self.length,1))
        neighborhoods = neighborhoods.T

        eyes = np.eye(self.length)
        tmp = eyes.copy()
        for i in range(1, n):
            index = [(i+j)%self.length for j in range(self.length)]
            tmp = tmp + eyes[index]
        logit = np.logical_xor(neighborhoods, tmp.T)
        tmp2 = np.zeros_like(neighborhoods, dtype=np.int)
        tmp2[logit] = 1
        neighborhoods = tmp2

        return neighborhoods

    def neighborhood_struct_n2(self, instance, n=3):
        assert len(instance) == self.length

        index = random.sample(range(self.length), n)
        for i in index:
            instance[i] = instance[i] ^ 1
        return instance

    # def find_optimum_local(self, instance):
    #     # 计算适应度函数
    #     value = -np.abs(self.bag_capacity - self.data @ instance)
    #     optimum = instance.T[np.where(value == value.min())]
    #     optimum = optimum.T
    #
    #     return optimum, value.min()

    def variable_neighborhood_search(self, population):
        # 遍历所有可行解
        local_optimums = np.array([[]] * self.length)
        for i in range(population.shape[1]):
            x = population[:, i]
            value_x = self.fitness_function(x)

            terminal = 6
            it = 0
            while True:
                l = 1
                while l <= 10:
                    # x_prime = self.neighborhood_struct_n(x, l)
                    # x_prime_prime, value = self.find_optimum_local(x_prime)
                    x_prime = self.neighborhood_struct_n2(x, l)
                    value_prime = self.fitness_function(x_prime)
                    if value_prime < value_x:
                        x = x_prime
                        value_x = value_prime
                    else:
                        l = l+1

                it = it + 1
                if it > terminal:
                    break
            local_optimums = np.c_[local_optimums, x]
        return local_optimums



    def main(self):
        """
        main函数,用来进化
        对当前种群依次进行选择、交叉并生成新一代种群，然后对新一代种群进行变异
        :return:
        """

        # 初始化种群
        init_population = self.initial_population()

        iter_plot = []
        f_plot = []
        iteration = 0

        f_set_plot = []

        # 记录最好的结果
        best_ans = []
        best_value = -np.inf
        # 进入迭代
        while iteration < self.iteration:
            # 经过选择后的父代
            parents = self.selection(init_population)
            # 产生下一代
            children = self.crossover(parents)
            # 变异
            mutation_children = self.mutation(children)

            # 变邻域搜索
            mutation_children = self.variable_neighborhood_search(mutation_children)

            # 更新种群
            # init_population = neight_best_population
            init_population = mutation_children

            f_set = []  # 求出每一步迭代的最大值
            for init in range(init_population.shape[1]):
                f_set_tem = self.fitness_function(init_population[:, init])
                f_set.append(f_set_tem)

            f_set = np.array(f_set)
            max_value = f_set.max()
            max_index = np.where(f_set == max_value)

            f_set_plot.append(max_value)

            iter_plot.append(iteration)
            iteration = iteration + 1
            print("第%s进化得如何******************************************" % iteration)
            f_average = self.fitness_average(init_population)

            f_plot.append(f_average)

            #             print(max_index)

            print(init_population[:, max_index[0][0]])
            print(max_value)

            if max_value > best_value:
                best_value = max_value
                best_ans = init_population[:, max_index[0][0]].copy()
        #             break
        # f_accumulation = f_accumulation + f
        #             # f_print = f_accumulation/(iteration + 1)
        #             # print(f_print)
        print('best answer: ', best_ans)
        print('best value: ', best_value)
        # self.plot_figure(iter_plot, f_plot, f_set_plot)
        return best_ans, best_value


def test_instance(instance):
    data = instance
    length = len(data)
    bag_capacity = (data.sum()) / 2

    g1 = GA_partition(number=300, iter_number=100, bag_capacity=bag_capacity, data=data)
    start = time.time()
    best_ans, best_value = g1.main()
    end = time.time()
    total_time = end - start
    print('消耗时间：', (end - start))
    return best_ans, best_value, total_time


def test_dataset():
    data_folder = './testdata'
    output_folder = './output'
    if os.path.exists(output_folder) is False:
        os.makedirs(output_folder)

    result_values = []
    result_time = []
    files = []
    for file_name in tqdm(os.listdir(data_folder)):
        result_file_name = file_name.split('.')[0]
        files.append(result_file_name)

        count_value = []
        count_time = []
        file_data = np.loadtxt(os.path.join(data_folder, file_name), dtype=np.int64, delimiter=',')
        for i in tqdm(range(file_data.shape[0])):
            try:
                instance = file_data[i, :]
            except:
                instance = file_data
            instance = np.array(instance)

            best_ans, best_value, total_time = test_instance(instance)
            count_time.append(total_time)
            count_value.append(best_value)

        result_values.append(sum(count_value)/len(count_value))
        result_time.append(sum(count_time)/len(count_value))

    files = np.array(files)
    result_values = np.array(result_values)
    result_time = np.array(result_time)

    final_results = np.vstack([files, result_values, result_time])
    final_results = final_results.T

    np.savetxt(os.path.join(output_folder, 'test_output.csv'), final_results, delimiter=',', fmt='%s')


if __name__ == '__main__':
    # instance = np.array([771, 121, 281, 854, 885, 734, 486, 1003, 83, 62])
    # test_instance(instance)

    test_dataset()

