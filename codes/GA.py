# -*- coding: utf-8 -*- 
# LAI Inversion: 用 MODIS 地表反射率数据和 ProSAIL 模型反演叶面积指数
# 这其实是我选的一个校级方法课【计算方法】的作业，我觉得写的还行，就拿过来反演了

# Author: phikun (201711051122@mail.bnu.edu.cn)
# Date: 2021.12.28

from typing import Callable, Collection, Tuple, List
from numpy.random import random, randint, uniform, choice
from tqdm import tqdm
import numpy as np


class individual:
    def __init__(self, x: np.ndarray):
        self.x = x

    def __eq__(self, other):
        return self.x == other.x and self.fitness == other.fitness

    def __lt__(self, other):
        """重载小于号，用于排序"""
        return self.fitness < other.fitness  # 把不等号反过来，就能实现自大到小排序；【最初的作业是要求最大值】

    def __str__(self):
        return f"individual: x = {self.x}, fitness = {self.fitness}"

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if name == "x":
            self.fitness = individual.fitness_func(self.x)  # 对 self.x 赋值时顺便计算它对应的适应度函数
            if np.isnan(self.fitness):
                self.fitness = 1E8  # 某些解算出来是 NaN、通过指定适应度函数非常大排除它们


class GA:
    def __init__(self, fitness: Callable, lb: Collection[float], rb: Collection[float], n_pop: int=40, p_mutation: float=0.1, max_iter: int=400):
        """
        遗传算法类的构造函数，在这里指定适应度函数、初始化种群
        :param fitness:     适应度函数，作为 individual 类的静态成员
        :param lb:          自变量 x 的左边界，Left Bound，需按顺序给定每个自变量的范围
        :param rb:          自变量 x 的右边界，Right Bound，需按顺序给定每个自变量的范围
        :param n_pop:       种群规模
        :param p_mutation:  变异概率
        :param max_iter:    最大迭代次数
        """
        individual.fitness_func = fitness
        self.__x_lb = lb
        self.__x_rb = rb
        self.__n_pop = n_pop
        self.__p_mutation = p_mutation
        self.__max_iter = max_iter
        self.__pop = self.__init_population()  # 初始化种群
        self.__cross_prob = np.arange(self.__n_pop, 0, -1) / np.arange(self.__n_pop, 0, -1).sum()  # 轮盘赌按逆位序置概率
    
    def __init_population(self) -> List[individual]:
        """初始化种群"""
        pop: List[individual] = []
        for _ in range(self.__n_pop):
            x = np.array([uniform(lb, rb) for (lb, rb) in zip(self.__x_lb, self.__x_rb)])  # 随机生成一个个体
            pop.append(individual(x))
        return pop

    def __selection(self) -> Tuple[int, int]:
        """因为种群始终是按最适应到最不适应有序排列的，所以按逆位序轮盘赌即可"""
        # ind_fitnesses = np.array([ind.fitness for ind in self.__pop])  # 每个个体的适应度函数
        # vmin = ind_fitnesses.min()
        # vmax = ind_fitnesses.max()
        # ind_fitnesses = (ind_fitnesses - vmin) / (vmax - vmin)         # 防止适应度函数是负数，影响概率值；做归一化
        # pvalues = ind_fitnesses / ind_fitnesses.sum()                  # 生成每个个体的轮盘赌的概率
        # (idx1, idx2) = choice(self.__n_pop, 2, p=pvalues)              # 用参数 p 指定每个个体被选择的概率
        (idx1, idx2) = choice(self.__n_pop, 2, p=self.__cross_prob)
        return (idx1, idx2)

    def __one_crossover(self) -> individual:
        """
        一次交叉互换，选择两个个体进行交叉互换，返回适应度函数高的一个个体
        交叉互换规则：按逆位序轮盘赌 2 个父代，1. 按 0.9/0.1 杂交; 2. 随机 k 个位置交叉; 3. 对交叉后的结果再 0.9/0.1 杂交
        对此步生成的 6 个新个体取最适应的
        """
        n = len(self.__x_lb)                                    # 可变参数个数
        (a, b) = self.__selection()
        (parent1, parent2) = (self.__pop[a], self.__pop[b])
        child1 = individual(0.9 * parent1.x + 0.1 * parent2.x)  # 因为 x 是 NumPy数组，所以便于直接做乘法；
        child2 = individual(0.1 * parent1.x + 0.9 * parent2.x)  # 这样要求子代的取值范围必在父代之间，避免了生成超出可行域的解
        
        if n == 1:
            return sorted([child1, child2])[0]  # 若只有1个变量，选择染色体的某一段交叉互换无意义，直接返回两个新个体中最适应的即可

        n_vars = randint(1, n)                           # 要交叉互换的参数个数
        var_indices = choice(range(n_vars), n_vars)      # 要交叉互换的变量
        (x0, x1) = (parent1.x.copy(), parent2.x.copy())  # 一定要 copy!
        for idx in var_indices:
            (x0[idx], x1[idx]) = (x1[idx], x0[idx])
        
        child3 = individual(x0)
        child4 = individual(x1)
        child5 = individual(0.9 * child3.x + 0.1 * child4.x)
        child6 = individual(0.1 * child3.x + 0.9 * child4.x)
        
        res = sorted([child1, child2, child3, child4, child5, child6])[0]
        return res

    def __crossover(self):
        """交配，生成新的 self.__n_pop 个个体的种群，并与此前的种群取最优 self.__n_pop 个"""
        new_pop = self.__pop + [self.__one_crossover() for _ in range(self.__n_pop)]  # 生成若干新个体，和此前的个体排在一起
        self.__pop = sorted(new_pop)[:self.__n_pop]  # 筛选最适应的前 self.__n_pop 个个体

    def __mutation(self):
        """变异，遍历每个个体，随机改变它所有参数中的一个；不保留最适应的，变了就是变了！"""
        n = len(self.__x_lb)  # 可变参数的数量

        for ind in self.__pop:
            if random() >= self.__p_mutation:
                continue
            if random() < 0.5:  # 只变一个参数
                x0 = ind.x.copy()
                idx = randint(1, n)                                    # 随机一个可变参数
                x0[idx] = uniform(self.__x_lb[idx], self.__x_rb[idx])  # 修改可变参数的值
                ind.x = x0                                             # 直接修改 individual.x，变异不可逆
            else:               # 所有参数都变化
                ind.x = np.array([uniform(lb, rb) for (lb, rb) in zip(self.__x_lb, self.__x_rb)])

    def implement(self) -> Tuple[np.ndarray, float, List[np.ndarray], List[float]]:
        xs = [None for _ in range(self.__max_iter + 1)]
        ys = [None for _ in range(self.__max_iter + 1)]

        self.__pop.sort()
        (xs[0], ys[0]) = (self.__pop[0].x, self.__pop[0].fitness)

        for it in tqdm(range(1, self.__max_iter + 1)):
            self.__crossover()  # 交叉互换，留下最适应的前 self.__n_pop 个
            self.__mutation()
            self.__pop.sort()
            (xs[it], ys[it]) = (self.__pop[0].x, self.__pop[0].fitness)  #  记录每轮迭代最优的 x 及其适应度函数
        
        best_y = sorted(ys)[0]         # 最优的 y，要排序后取首个是因为可能越大越好或越小越好，这交给 indivudal.__lt__ 决定
        best_x = xs[ys.index(best_y)]  # 对应最优 y 的最优 x

        return (best_x, best_y, xs, ys)
