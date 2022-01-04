# -*- coding: utf-8 -*-
# LAI Inversion: 用 MODIS 地表反射率数据和 ProSAIL 模型反演叶面积指数
# 优化器：调用 scikit-opt 或自己手动实现的遗传算法优化代价函数：【还是 scikit-opt 实现的遗传算法效率更高】
# 静态参数从 ../data/optimize_information.json 读入，动态参数从 ../data/optimize_params.json 中读入
# 【因为 MATLAB Runtime 和 gdal 冲突，所以把优化的部分单甩出一个进程】

# Author: phikun (201711051122@mail.bnu.edu.cn)
# Date: 2022.01.03

from prosaillib import pyprosail
from sko.tools import set_run_mode  # scikit-opt 加速魔法
from sko.GA import GA as skoGA      # scikit-opt 库实现的遗传算法
from numpy.random import seed
from time import perf_counter
import numpy as np
import json

from GA import GA as myGA       # 我自己手动实现的遗传算法


# 代价函数类，覆写 __call__ 方法实现类似于函数调用的特性
class loss_function:
    def __init__(self):
        static_info_fname = "../data/optimize_information.json"  # 静态参数，包括各波段的不确定性、各参数的默认值和取值范围
        dynamic_params_fname = "../data/optimize_params.json"    # 动态参数，包括要使用的波段、各波段的反射率、先验知识、观测几何

        self.__get_dynamic_info(dynamic_params_fname)
        self.__get_static_info(static_info_fname)
        self.__lai_index = self.__model_params.index("LAI")  # LAI 在模型参数中的位置，用于获取 LAI 的值构造代价函数 
        self.__prosail = pyprosail()  # 默认是按光谱响应函数加权

    def __call__(self, x: np.ndarray) -> np.ndarray:
        assert len(self.__model_params) == len(x)

        param_dic = dict(zip(self.__model_params, list(map(float, x))))
        param_dic.update(self.__static_param_dic)  # 把所有静态参数和可变参数合并，作为 ProSAIL 模型的输入
        (band_ref, _) = self.__prosail.run(**param_dic)

        lai = x[self.__lai_index]  # ; print(f"lai = {lai}")
        values = np.zeros(len(self.__bands) + 1)
        values[0] = (lai - self.__lai_mean) ** 2 / self.__lai_std  # 损失函数中先验知识项
        for (idx, band) in enumerate(self.__bands, 1):
            values[idx] = (band_ref[band] - self.__ref[band]) ** 2 / self.__band_sigma[band]  # 各波段项，以不确定度的倒数作为权重
        
        res = values.sum()  # ; print(f"loss = {values}")
        res = 1E8 if np.isnan(res) else res  # 防止出现 NaN 值影响结果
        return res

    def __get_dynamic_info(self, fname: str):
        """读入主进程处理后写入文件的反演参数"""
        with open(fname, "r") as fin:
            dic = json.load(fin)
        
        self.__bands = dic["bands"]          # 反演要用到的波段
        self.__ref = dic["ref"]              # 各波段的反射率，字典形式：{"b01": 0.2, "b02": 0.4}
        self.__model_params = dic["params"]  # 用到的 ProSAIL 模型参数，列表形式，优化时需要按顺序指定
        self.__lai_mean = dic["LAI-Mean"]    # MODIS 二十年的 LAI 均值，先验知识
        self.__lai_std = dic["LAI-Std"]      # MODIS 二十年的 LAI 标准差，先验知识
        (self.__tts, self.__tto, self.__psi) = (dic["tts"], dic["tto"], dic["psi"])  # 观测几何，常数

    def __get_static_info(self, fname: str):
        """根据用到的波段和模型参数获取静态参数，并生成初始点、下端点和上端点"""
        with open(fname, "r") as fin:
            dic = json.load(fin)
        (sigma, params) = (dic["Uncertainty"], dic["ModelParams"])  # 各波段的不确定性、模型参数
        
        self.__band_sigma = dict(zip(self.__bands, [sigma[band] for band in self.__bands]))  # 用到的波段的不确定性，其倒数作为权重
        
        static_params = params.keys() - set(self.__model_params)  # 所有可变参数与用到的模型参数的差集即是不变参数
        self.__static_param_dic = dict(zip(static_params, [params[param]["default"] for param in static_params]))  # 对不变参数取默认值
        self.__static_param_dic.update({"tts": self.__tts, "tto": self.__tto, "psi": self.__psi})                  # 并加入观测几何信息

        self.x0 = [params[param]["default"] for param in self.__model_params]  # 以参数的默认值作为初始搜索点
        self.lb = [params[param]["lb"] for param in self.__model_params]       # 所有用到的 ProSAIL 参数取值范围的下端点 lower bound
        self.ub = [params[param]["ub"] for param in self.__model_params]       # 所有用到的 ProSAIL 参数取值范围的上端点 upper bound


if __name__ == "__main__":
    print("Hello World!")
    seed(42)
    st = perf_counter()

    func = loss_function()
    (x0, lb, ub) = (func.x0, func.lb, func.ub)
    n_dim = len(x0)

    set_run_mode(func, "cached")  # 缓存加速，遗传算法后期动的少，这应该很管用

    print(lb)
    print(ub)
    (n_pop, max_iter, eps) = (50, 200, 1E-7)  # scikit-opt 实现的遗传算法的默认参数

    ga = skoGA(func, n_dim, size_pop=n_pop, max_iter=max_iter, lb=lb, ub=ub, precision=eps)  # skoGA
    (best_x, best_y) = ga.run()

    # ga = myGA(func, lb, ub, n_pop=n_pop, p_mutation=0.2, max_iter=max_iter)  # myGA，效率太低了，所以就用 scikit-opt 实现的遗传算法
    # (best_x, best_y, xs, ys) = ga.implement()
    print("best_x:", best_x)
    print("best_y:", best_y)

    et = perf_counter()
    print(f"time = {round(et - st)}s.")

    print("Finished.")
