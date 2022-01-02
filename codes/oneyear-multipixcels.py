# -*- coding: utf-8 -*-
# LAI Inversion: 用 MODIS 地表反射率数据和 ProSAIL 模型反演叶面积指数
# 作业一：一期、多个像元各自反演 LAI，并与 SPOT 高分辨率合成到 500m 分辨率的参考值比较

# Author: phikun (201711051122@mail.bnu.edu.cn)
# Date: 2022.01.02

from typing import Collection, Tuple, List, Dict
from matplotlib import pyplot as plt
from glob import glob
import geopandas as gpd
import pandas as pd
import numpy as np
import json
import os

from converter import raster2fishnet
import utility as util


# 一期、多个像元的反演器，包含了需要像元的行列号索引，逐个像元进行反演
class OyMpInvertor:
    def __init__(self, yearday: str, indices=None, bands: Collection[str]=None, model_params: Collection[str]=None):
        """
        构造函数
        :param yearday:      待反演的年份和日序，例如 2000049 表示 2000 年第 49 天
        :param indices:      待反演像元的索引，若为 None 则指定是该区域内所有完全落入 SPOT 影像范围内的栅格
        :param bands:        反演用到的波段，若为 None 则表示使用全部 b01~b07 共 7 个波段
        :param model_params: 要用到的 ProSAIL 模型参数，若为 None 则使用所有参数
        """
        self.__lai_path = "../data/MODIS LAI/"  # MODIS LAI 的路径
        self.__ref_path = "../data/MODIS SR/"   # MODIS 地表反射率的路径
        self.__spot_file = "../data/ZhangBei/SPOTZhangbei20020809TOA_VarBioMaps/SPOTZhangbei20020809TOA_VarBioMaps.bil"  # SPOT 的高分辨率反演结果，其第一个波段恰好是有效叶面积指数
        self.__yearday = yearday

        self.__bands = bands if bands is not None else ["b01", "b02", "b03", "b04", "b05", "b06", "b07"]
        self.__model_params = model_params if model_params is not None else ["N", "Cab", "Car", "Cbrown", "Cw", "Cm", "LIDFa", "LIDFb", "TypeLidf", "LAI", "hspot", "tts", "tto", "psi", "psoil"]
        self.__opt_param_file = "../data/optimize_params.json"  # 代价函数优化的用到参数，因为 gdal 和 MATLAB 运行时冲突，所以单开一个进程做优化

        if indices is not None:
            self.__indices = indices
        else:
            rows = [r for r in range(7, 13) for _ in range(29, 37)]
            cols = [c for _ in range(7, 13) for c in range(29, 37)]
            self.__indices = list(zip(rows, cols))

    def __get_ref_lai(self) -> Dict[Tuple[int, int], float]:
        """用 SPOT 的结果合成参考值，"""
        # Step1: 用 self.__indice 索引，获取需要的 MODIS 栅格，
        modis_lai_raster = glob(f"{self.__lai_path}*{self.__yearday}*Lai_500m.tif")[0]
        modis_gdf = raster2fishnet(modis_lai_raster)
        rc = list(zip(modis_gdf["Row"], modis_gdf["Column"]))
        modis_gdf.index = rc                          # 用每个像元的行列号作为索引
        modis_gdf = modis_gdf.loc[self.__indices, :]  # 用self.__indices 从索引中取值，这里关心渔网的空间位置

        # Step2: 获取 SPOT 栅格，并投影到与 MODIS 的坐标系一致
        spot_gdf = raster2fishnet(self.__spot_file).to_crs(modis_gdf.crs)
        spot_gdf["Value"] = spot_gdf["Value"].astype(float) / 1000.0  # SPOT 反演产品的DN值除以 1000 才是需要的叶面积指数
        
        # Step3: 将 MODIS 和 SPOT 栅格叠加，统计完全包含在某个 MODIS 像元中的 SPOT LAI 的均值
        crossover = gpd.sjoin(modis_gdf, spot_gdf, how="left", op="contains")   # contains 做空间叠加
        means = crossover.reset_index().groupby("index")["Value_right"].mean()  # 计算落入某个 MODIS 像元的所有 SPOT 像元的均值
        ref_lai = dict(means)
        return ref_lai        

    def __get_lai_mean_and_std(self, row: int, col: int) -> Tuple[float, float]:
        """获取一个像元同期 20 年的均值和标准差，构造代价函数中先验信息项"""
        (year, day) = (self.__yearday[:4], self.__yearday[4:])
        lais: float = []

        for y in range(2000, 2021):
            if str(y) == year:
                continue
            yd = f"{y}{day}"
            fnames= glob(f"{self.__lai_path}*{yd}*Lai_500m.tif")
            if len(fnames) == 0:
                print(f"    cannot find LAI file of yearday: {yd}.")
                continue
            (data, _) = util.read_geotiff(fnames[0])
            lais.append(data[row, col] / 10.0)  # MODIS LAI 产品的 scale factor 是 10
        
        lais = np.array(lais)
        return (lais.mean(), lais.std())

    def __get_band_reflectance(self, row: int, col: int) -> Dict[str, float]:
        """获取波段反射率"""
        dic: Dict[str, float] = {}
        for band in self.__bands:
            fname = glob(f"{self.__ref_path}*{self.__yearday}*{band}.tif")[0]
            (data, _) = util.read_geotiff(fname)
            dic[band] = data[row, col] / 1E4  # MODIS 反射率产品的 scale factor 是 0.0001
        return dic

    def __invert_one_pixcel(self, row: int, col: int):
        """给出像元的行列号，反演一个像元的值"""
        # Step1: 获取 LAI 先验信息和反射率信息，写入 json 文件
        (lai_mean, lai_std) = self.__get_lai_mean_and_std(row, col)
        band_ref = self.__get_band_reflectance(row, col)
        opt_params = {"bands"   :  self.__bands,        # 用到的波段
                      "ref"     :  band_ref,            # 每个波段的额反射率
                      "params"  : self.__model_params,  # ProSAIL 模型参数
                      "LAI-Mean": lai_mean,             # 先验知识：LAI 均值
                      "LAI-Std" : lai_std}              # 先验知识：LAI 标准差
        with open(self.__opt_param_file, "w") as fout:
            json.dump(opt_params, fout)

        # Step2: 代价函数最优化

    def run(self):
        # Step0: 用 SPOT 的结果合成 LAI 参考值
        # ref_lai = self.__get_ref_lai()

        # Step1: 逐像元反演
        for (row, col) in self.__indices:
            self.__invert_one_pixcel(row, col)
            break


if __name__ == "__main__":
    print("Hello World!")

    yearday = "2002225"     # 8 月 8~10 日对应第 230~232 天，在第 225~232 天的合成产品中
    bands = ["b01", "b02"]  # 只用红和近红外的反射率
    model_params = ["LAI", "LIDFa", "Cab", "Cm", "N"]

    inverter = OyMpInvertor(yearday, bands=bands, model_params=model_params)
    inverter.run()

    print("Finished.")
