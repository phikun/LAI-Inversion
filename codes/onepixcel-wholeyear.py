# -*- coding: utf-8 -*-
# LAI Inversion: 用 MODIS 地表反射率数据和 ProSAIL 模型反演叶面积指数
# 作业二：反演站点对应像元任意一年的LAI，并与 MODIS LAI 对比分析

# Author: phikun (201711051122@mail.bnu.edu.cn)
# Date: 2022.01.05

from typing import Collection, Tuple, List, Dict
from math import pi, sin, cos, asin
from glob import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
import logging
import json
import os

import utility as util


def calc_tts(daynum: int, lat: float, time: float, degree: bool=True):
    """
    计算太阳高度角，和网上算的不太一样，不过总体趋势应该是对的；就这样吧
    :param daynum: DAY NUMber，日序，从 1 月 1 日起算
    :param lay:    纬度，北纬为正
    :param time:   时角，正午是 0，上午为负、下午为正
    :param degree: 输入角的单位，度（默认）或弧度
    """
    if degree is True:
        time = time * pi / 180.0
        lat = lat * pi / 180.0
    
    day_theta = (2.0 * pi * (daynum - 80) / 365.0)  # 日序角，从春分日起算；春分日大概是 3 月 21 日：31+28+21=80
    tropic = (23 + 26 / 60) * pi / 180.0            # 南北回归线范围，23 度 26 分
    delta = asin(sin(day_theta) * sin(tropic))      # 太阳赤纬
    hs = asin(sin(lat) * sin(delta) + cos(lat) * cos(delta) * cos(time))  # 太阳高度角
    hs = 90.0 - hs * 180.0 / pi  # 把太阳高度角转成太阳天顶角
    return hs


class OpWyInvertor:
    def __init__(self, year: str, output_file: str, lat: float, time: float, index: Tuple[int, int], bands: Collection[str]=None, model_params: Collection[str]=None):
        """
        构造函数
        :param year:         待反演的年份
        :param output_file:  结果文件，把反演结果和同期 MODIS LAI 产品写到文件里 
        :param lat:          像元纬度，十进制度，北纬为正，用于计算太阳天顶角
        :param time:         时角，十进制度，正午是 0，上午为负、下午为正，用于计算太阳天顶角
        :param index:        待反演像元的索引
        :param bands:        反演用到的波段，若为 None 则表示使用全部 b01~b07 共 7 个波段
        :param model_params: 要用到的 ProSAIL 模型参数，若为 None 则使用所有参数
        """
        self.__lai_path = "../data/MODIS LAI/"  # MODIS LAI 的路径
        self.__ref_path = "../data/MODIS SR/"   # MODIS 地表反射率的路径
        self.__lat = lat
        self.__year = year
        self.__time = time
        self.__index = index
        self.__output_file = output_file
        self.__obs_geom = {"tts": None, "tto": 0.0, "psi": 0.0}  # 观测几何，太阳天顶角每次算

        self.__bands = bands if bands is not None else ["b01", "b02", "b03", "b04", "b05", "b06", "b07"]
        self.__model_params = model_params if model_params is not None else ["N", "Cab", "Cw", "Cm", "LIDFa", "LAI", "hspot", "psoil"]

        (self.__logger, self.__fh, self.__ch) = OpWyInvertor.__init_logger()

    def __get_modis_lai(self) -> Dict[int, float]:
        """获取一年内每一天的 MODIS LAI 产品的值"""
        daynums = list(range(1, 365, 8))  # 日序，八天合成
        lais: List[float] = []

        for daynum in daynums:
            fname = glob(f"{self.__lai_path}*{self.__year}{daynum:03d}*Lai_500m.tif")[0]
            (data, _) = util.read_geotiff(fname)
            lai = data[self.__index[0], self.__index[1]] / 10.0  # MODIS LAI 产品的尺度银子是 10
            lais.append(lai)
        
        return dict(zip(daynums, lais))

    def __get_lai_mean_and_std(self, daynum: int) -> Tuple[float, float]:
        """获取一个像元同期 20 年的均值和标准差，构造代价函数中先验信息项"""
        lais = []
        for year in range(2000, 2021):
            fnames = glob(f"{self.__lai_path}*{year}{daynum:03d}*Lai_500m.tif")
            if len(fnames) == 0:
                print(f"    cannot find LAI file of yearday: {year}{daynum:03d}.")
                continue
            (data, _) = util.read_geotiff(fnames[0])
            lai = data[self.__index[0], self.__index[1]] / 10.0
            lais.append(lai)
        
        lais = np.array(lais)
        return (lais.mean(), lais.std())

    def __get_band_reflectance(self, daynum: int) -> Dict[str, float]:
        """获取波段反射率"""
        dic: Dict[str, float] = {}
        for band in self.__bands:
            fname = glob(f"{self.__ref_path}*{self.__year}{daynum:03d}*{band}.tif")[0]
            (data, _) = util.read_geotiff(fname)
            ref = data[self.__index[0], self.__index[1]] / 1E4  # MODIS 反射率产品的 scale factor 是 0.0001
            dic[band] = ref
        return dic

    def __invert_one_day(self, daynum: int) -> float:
        """给出日序，反演一天的 LAI；这与作业一中相似功能的函数很像，可以进一步抽象！"""
        # Step1: 获取 LAI 先验信息和反射率信息，写入 json 文件
        (lai_mean, lai_std) = self.__get_lai_mean_and_std(daynum)
        band_ref = self.__get_band_reflectance(daynum)
        self.__obs_geom["tts"] = calc_tts(daynum, self.__lat, self.__time, degree=True)
        opt_params = {"bands"   : self.__bands,         # 用到的波段
                      "ref"     : band_ref,             # 每个波段的额反射率
                      "params"  : self.__model_params,  # ProSAIL 模型参数
                      "LAI-Mean": lai_mean,             # 先验知识：LAI 均值
                      "LAI-Std" : lai_std}              # 先验知识：LAI 标准差
        opt_params.update(self.__obs_geom)

        opt_param_file = "../data/optimize_params.json"  # 代价函数优化的用到参数，因为 gdal 和 MATLAB 运行时冲突，所以单开一个进程做优化
        with open(opt_param_file, "w") as fout:
            json.dump(opt_params, fout)                  # 把光谱参数、模型参数和观测几何一并写入文件中

        # Step2: 代价函数最优化
        cmd = "python optimize.py"                      # 单开一个进程做优化
        output_file = "../data/optimize_results.json"   # 由 optimize.py 写出的优化结果文件，从中读入优化信息
        return_value = os.system(cmd)

        with open(output_file, "r") as fin:
            dic = json.load(fin)
        (best_x, best_y, time) = (dic["best_x"], dic["best_y"], dic["time"])
    
        lai_index = self.__model_params.index("LAI")  # LAI 在模型参数中的位置
        best_lai = best_x[lai_index]
        dic.update({"daynum": daynum, "best_lai": best_lai})

        # Step3: 把优化结果写入文件
        self.__logger.info(f"Finished pixcel in day ({daynum}), with return value: {return_value}, best_lai = {best_lai}, best_y = {best_y}, time = {time}.")
        middle_results = "./prob2-middle-results.txt"
        with open(middle_results, "a") as fout:
            fout.write(f"{json.dumps(dic)}\n")
        return best_lai

    def run(self):
        # Step1: 逐日反演
        inv_lai = [self.__invert_one_day(daynum) for daynum in tqdm(range(1, 365, 8))]

        # Step2: 当期 MODIS LAI 产品的像元值
        modis_lai = self.__get_modis_lai()

        df = pd.DataFrame({"Inversion": inv_lai, "MODIS": list(modis_lai.values())}, index=range(1, 365, 8))
        df.to_excel(self.__output_file, engine="openpyxl", index=True)

    @staticmethod
    def __init_logger():
        """初始化日志"""
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler("./prob2-logging.log")  # 输出至文件
        ch = logging.StreamHandler()  # 输出至控制台
        fomatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(fomatter)
        ch.setFormatter(fomatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
        return (logger, fh, ch)  # 把这两个handle都输出，一个区域结束之后断开handle，防止输出数量不断增加


if __name__ == "__main__":
    print("Hello World!")

    year = "2020"
    index = (10, 32)  # 采样点所在栅格的行列号
    (lat, time) = (41.278889, -30.0)  # 纬度、时角（卫星过境时间大概是上午十点，所以时角是 -30）
    output_file = "../results/Question2.xlsx"
    bands = ["b01", "b02"]
    model_params = ["LAI", "LIDFa", "Cab", "Cm", "N"]
    
    invertor = OpWyInvertor(year, output_file, lat, time, index, bands=bands, model_params=model_params)
    invertor.run()

    print("Finished.")
