# -*- coding: utf-8 -*-
# LAI Inversion: 用 MODIS 地表反射率数据和 ProSAIL 模型反演叶面积指数
# 反演器的基类，指定两个问题中共有的做法

# Author: phikun (201711051122@mail.bnu.edu.cn)
# Date: 2021.01.11

from typing import Collection, Tuple, Dict
from abc import abstractmethod
from glob import glob
import numpy as np
import logging
import json
import os
import re

import utility as util


# 反演器的基类
class invertor:
    def __init__(self, bands: Collection[str]=None, model_params: Collection[str]=None, log_fname: str="./logging.log", 
                       opt_param_file: str="../data/optimize_params.json", opt_res_file: str="../data/optimize_results.json", middle_res: str="./base-middle-result.txt"):
        """
        构造函数
        :param bands:          用于反演的波段，形如 b01, b02, ...，若不指定则是 MODIS 前 7 个波段
        :param model_params:   用到的 ProSAIL 模型参数，若不指定则默认是所有合法的参数
        :param log_fname:      logger 的文件名
        :param opt_param_file: 把优化的代价函数中要用到的参数写到文件里，然后调用 optimize.py 进行优化
        :param opt_res_file:   优化进程 optimize.py 的输出文件，从中读入反演结果
        :param middle_res:     把反演每个栅格时的全部输出记录依次写到一个文件里
        """
        all_params = ["N", "Cab", "Cw", "Cm", "LIDFa", "LAI", "hspot", "psoil"]  # 所有合法的模型参数

        self._bands = bands if bands is not None else ["b01", "b02", "b03", "b04", "b05", "b06", "b07"]
        self._model_params = model_params if model_params is not None else all_params

        assert all(map(lambda band: re.match(r"b\d{2}", band), self._bands))    # 检查波段名称是否合法
        assert all(map(lambda param: param in all_params, self._model_params))  # 检查模型参数是否合法

        self._lai_path = "../data/MODIS LAI/"  # MODIS LAI 的路径
        self._ref_path = "../data/MODIS SR/"   # MODIS 地表反射率的路径
        self._obs_geom = {"tts": None, "tto": None, "psi": None}  # 观测几何，到子类指定

        self._opt_param_file = opt_param_file
        self._opt_result_file = opt_res_file
        self._middle_result_file = middle_res
        (self._logger, self._fh, self._ch) = invertor.__init_logger(log_fname)

    def __del__(self):
        """析构函数，移除 logger 的句柄"""
        self._logger.removeHandler(self._fh)
        self._logger.removeHandler(self._ch)

    def _get_lai_mean_and_std(self, year: int, daynum: int, row: int, col: int) -> Tuple[float, float]:
        """
        获取 LAI 的均值和标准差，访问 self.__lai_path 中二十年同期的 LAI 产品，构造先验知识
        :param year:   年份，整数；用于去掉待反演当年的数据
        :param daynum: 日序，整数
        :param row:    待反演栅格的行号
        :param col:    待反演栅格的列号
        """
        lais: float = []

        for y in list(set(range(2000, 2021)) - {year}):  # 去掉待反演当年的数据
            yd = f"{y}{daynum:03d}"
            fnames = glob(f"{self._lai_path}*{yd}*Lai_500m.tif")  # 获取此前每年的 LAI 产品名称
            if len(fnames) == 0:
                self._logger.info(f"    cannot find LAI file of yearday: {yd}.")
                continue
            (data, _) = util.read_geotiff(fnames[0])
            lais.append(data[row, col] / 10.0)  # MODIS LAI 产品的 scale factor 是 10
        
        lais = np.array(lais)
        return (lais.mean(), lais.std())

    def _get_band_reflectance(self, year: int, daynum: int, row: int, col: int) -> Dict[str, float]:
        """获取当前年份和日序的、模型中用到的波段的 MODIS 反射率值"""
        dic: Dict[str, float] = {}
        for band in self._bands:
            fname = glob(f"{self._ref_path}*{year}{daynum:03d}*{band}.tif")[0]
            (data, _) = util.read_geotiff(fname)
            dic[band] = data[row, col] / 1E4  # MODIS 反射率产品的 scale factor 是 0.0001
        return dic

    def _calc_and_write_opt_param_to_file(self, year: int, daynum: int, row: int, col: int):
        """获取反演要用到的参数（LAI 先验知识、各波段的反射率），并写入 self._opt_param_file 文件"""
        (lai_mean, lai_std) = self._get_lai_mean_and_std(year, daynum, row, col)
        band_ref = self._get_band_reflectance(year, daynum, row, col)
        opt_params = {"bands"   : self._bands,          # 用到的波段
                      "ref"     : band_ref,             # 每个波段的额反射率
                      "params"  : self._model_params,   # ProSAIL 模型参数
                      "LAI-Mean": lai_mean,             # 先验知识：LAI 均值
                      "LAI-Std" : lai_std}              # 先验知识：LAI 标准差
        opt_params.update(self._obs_geom)               # 注意在函数外面指定和修改观测几何！

        with open(self._opt_param_file, "w") as fout:
            json.dump(opt_params, fout)

    def _read_opt_result_from_file(self) -> dict:
        """从 self._opt_result_file 中读入反演结果，以字典形式返回"""
        with open(self._opt_result_file, "r") as fin:
            dic = json.load(fin)
        
        lai_index = self._model_params.index("LAI")  # LAI 在模型参数中的位置
        best_lai = dic["best_x"][lai_index]
        dic.update({"best_lai": best_lai})
        return dic

    def _base_invert_one_pixcel(self, year: int, daynum: int, row: int, col: int) -> float:
        """反演一个栅格的值，baseline，可以在继承时加入预处理和后处理；返回优化得到的最优 LAI"""
        self._calc_and_write_opt_param_to_file(year, daynum, row, col)  # 计算并将需要用到的反演参数写入文件

        cmd = f"python optimize.py -inp {self._opt_param_file} -out {self._opt_result_file}"  # 开一个 Python 进程做优化
        return_value = os.system(cmd)

        dic = self._read_opt_result_from_file()            # 从文件中读入反演结果
        dic.update({"row": row, "column": col})
        with open(self._middle_result_file, "a") as fout:  # 并将全部反演结果写入文件
            fout.write(f"{json.dumps(dic)}\n")
        
        self._logger.info(f"Finished pixcel at ({year}, {daynum}, {row}, {col}), with return value: {return_value}, best_lai = {dic['best_lai']}, best_y = {dic['best_y']}, time = {dic['time']}.")
        return dic["best_lai"]

    @abstractmethod
    def run(self):
        raise RuntimeError("Call abstract method: invertor.run")

    @staticmethod
    def __init_logger(fname: str):
        """初始化日志"""
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(fname)  # 输出至文件
        ch = logging.StreamHandler()     # 输出至控制台
        fomatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(fomatter)
        ch.setFormatter(fomatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
        return (logger, fh, ch)  # 把这两个handle都输出，一个区域结束之后断开handle，防止输出数量不断增加
