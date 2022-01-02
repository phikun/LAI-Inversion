# -*- coding: utf-8 -*-
# LAI Inversion: 用 MODIS 地表反射率数据和 ProSAIL 模型反演叶面积指数
# 作业一：一期、多个像元各自反演 LAI，并与 SPOT 高分辨率合成到 500m 分辨率的参考值比较

# Author: phikun (201711051122@mail.bnu.edu.cn)
# Date: 2022.01.02

from typing import Tuple, Dict
from matplotlib import pyplot as plt
from glob import glob
import geopandas as gpd
import pandas as pd

from converter import raster2fishnet
import utility as util


# 一期、多个像元的反演器，包含了需要像元的行列号索引，逐个像元进行反演
class OyMpInvertor:
    def __init__(self, yearday: str, indices=None):
        """
        构造函数
        :param yearday: 待反演的年份和日序，例如 2000049 表示 2000 年第 49 天
        :param indices: 待反演像元的索引，若为 None 则指定是该区域内所有完全落入 SPOT 影像范围内的栅格
        """
        self.__lai_path = "../data/MODIS LAI/"  # MODIS LAI 的路径
        self.__ref_path = "../data/MODIS SR/"   # MODIS 地表反射率的路径
        self.__spot_file = "../data/ZhangBei/SPOTZhangbei20020809TOA_VarBioMaps/SPOTZhangbei20020809TOA_VarBioMaps.bil"  # SPOT 的高分辨率反演结果，其第一个波段恰好是有效叶面积指数
        self.__yearday = yearday

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

    def __invert_one_pixcel(self, row: int, col: int):
        """给出像元的行列号，反演一个像元的值"""
        # Step1: 获取先验信息，构造代价函数
        # Step2: 代价函数最优化

    def run(self):
        # Step0: 用 SPOT 的结果合成 LAI 参考值
        # ref_lai = self.__get_ref_lai()

        # Step1: 逐像元反演
        for (row, col) in self.__indices:
            pass


if __name__ == "__main__":
    print("Hello World!")

    yearday = "2002225"  # 8 月 8~10 日对应第 230~232 天，在第 225~232 天的合成产品中
    inverter = OyMpInvertor(yearday)
    inverter.run()

    print("Finished.")
