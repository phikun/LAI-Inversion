# -*- coding: utf-8 -*-
# LAI Inversion: 用 MODIS 地表反射率数据和 ProSAIL 模型反演叶面积指数
# 作业一：一期、多个像元各自反演 LAI，并与 SPOT 高分辨率合成到 500m 分辨率的参考值比较

# Author: phikun (201711051122@mail.bnu.edu.cn)
# Date: 2022.01.02

from typing import Collection, Tuple, List, Dict
from tqdm import tqdm
from glob import glob
import geopandas as gpd
import pandas as pd
import numpy as np
import logging
import json
import os

from converter import raster2fishnet
import utility as util


# 一期、多个像元的反演器，包含了需要像元的行列号索引，逐个像元进行反演
class OyMpInvertor:
    def __init__(self, yearday: str, output_file, tts: float, tto: float, psi: float, indices=None, bands: Collection[str]=None, model_params: Collection[str]=None):
        """
        构造函数
        :param yearday:       待反演的年份和日序，例如 2000049 表示 2000 年第 49 天
        :param output_file:   结果文件，干脆把每个像元反演得到的 LAI、SPOT 向上聚合的 LAI 和 MODIS 同期 LAI 产品的值写到文件里
        :param tts, tto, psi: 太阳天顶角、观测天顶角、相对方位角，直接在 ProSAIL 模型中指定
        :param indices:       待反演像元的索引，若为 None 则指定是该区域内所有完全落入 SPOT 影像范围内的栅格
        :param bands:         反演用到的波段，若为 None 则表示使用全部 b01~b07 共 7 个波段
        :param model_params:  要用到的 ProSAIL 模型参数，若为 None 则使用所有参数
        """
        self.__lai_path = "../data/MODIS LAI/"  # MODIS LAI 的路径
        self.__ref_path = "../data/MODIS SR/"   # MODIS 地表反射率的路径
        self.__spot_file = "../data/ZhangBei/SPOTZhangbei20020809TOA_VarBioMaps/SPOTZhangbei20020809TOA_VarBioMaps.bil"  # SPOT 的高分辨率反演结果，其第一个波段恰好是有效叶面积指数
        self.__yearday = yearday
        self.__output_file = output_file
        self.__obs_geom = {"tts": tts, "tto": tto, "psi": psi}  # 观测几何

        self.__bands = bands if bands is not None else ["b01", "b02", "b03", "b04", "b05", "b06", "b07"]
        self.__model_params = model_params if model_params is not None else ["N", "Cab", "Cw", "Cm", "LIDFa", "LAI", "hspot", "psoil"]
        self.__middle_results = "./middle_results.txt"  # 中间结果，把每个像元反演的 best_x, best_y, y_history 等指标逐行写到这里
        (self.__logger, self.__fh, self.__ch) = OyMpInvertor.__init_logger()

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

    def __get_modis_lai(self) -> Dict[Tuple[int, int], float]:
        """获取当期 MODIS LAI 产品各像元的值，用于与反演结果比较"""
        fname = glob(f"{self.__lai_path}*{self.__yearday}*Lai_500m.tif")[0]
        (data, _) = util.read_geotiff(fname)
        values = [data[row, col] / 10.0 for (row, col) in self.__indices]
        res = dict(zip(self.__indices, values))
        return res

    def __get_lai_mean_and_std(self, row: int, col: int) -> Tuple[float, float]:
        """获取一个像元同期 20 年的均值和标准差，构造代价函数中先验信息项"""
        (year, day) = (self.__yearday[:4], self.__yearday[4:])
        lais: float = []

        for y in range(2000, 2021):
            if str(y) == year:
                continue
            yd = f"{y}{day}"
            fnames = glob(f"{self.__lai_path}*{yd}*Lai_500m.tif")
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

    def __invert_one_pixcel(self, row: int, col: int) -> float:
        """给出像元的行列号，反演一个像元的 LAI 值；"""
        # Step1: 获取 LAI 先验信息和反射率信息，写入 json 文件
        (lai_mean, lai_std) = self.__get_lai_mean_and_std(row, col)
        band_ref = self.__get_band_reflectance(row, col)
        opt_params = {"bands"   :  self.__bands,        # 用到的波段
                      "ref"     :  band_ref,            # 每个波段的额反射率
                      "params"  : self.__model_params,  # ProSAIL 模型参数
                      "LAI-Mean": lai_mean,             # 先验知识：LAI 均值
                      "LAI-Std" : lai_std}              # 先验知识：LAI 标准差
        opt_params.update(self.__obs_geom)

        opt_param_file = "../data/optimize_params.json"  # 代价函数优化的用到参数，因为 gdal 和 MATLAB 运行时冲突，所以单开一个进程做优化
        with open(opt_param_file, "w") as fout:
            json.dump(opt_params, fout)                  # 把光谱参数、模型参数和观测几何一并写入文件中

        # Step2: 代价函数最优化
        cmd = "python optimize.py"                      # 单开一个进程做优化
        output_file = "../data/optimitze_results.json"  # 由 optimize.py 写出的优化结果文件，从中读入优化信息
        return_value = os.system(cmd)

        with open(output_file, "r") as fin:
            dic = json.load(fin)
        (best_x, best_y, time) = (dic["best_x"], dic["best_y"], dic["time"])
    
        lai_index = self.__model_params.index("LAI")  # LAI 在模型参数中的位置
        best_lai = best_x[lai_index]
        dic.update({"row": row, "col": col, "best_lai": best_lai})

        # Step3: 把优化结果写入文件
        self.__logger.info(f"Finished pixcel at ({row}, {col}), with return value: {return_value}, best_lai = {best_lai}, best_y = {best_y}, time = {time}.")
        with open(self.__middle_results, "a") as fout:
            fout.write(f"{json.dumps(dic)}\n")
        return best_lai

    def run(self):
        # Step1: 逐像元反演，很慢
        lai_values = [self.__invert_one_pixcel(row, col) for (row, col) in tqdm(self.__indices)]
        inv_lai = dict(self.__indices, lai_values)
        
        # Step2: 用 SPOT 的结果合成 LAI 参考值
        ref_lai = self.__get_ref_lai()

        # Step3: 当期 MODIS LAI 产品的像元值
        modis_lai = self.__get_modis_lai()

        df = pd.DataFrame({"Inversion": inv_lai, "SPOT": ref_lai, "MODIS": modis_lai}, index=self.__indices)  # 先跑起来试试
        df.to_excel(self.__output_file, engine="openpyxl", index=True)
    
    @staticmethod
    def __init_logger():
        """初始化日志"""
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler("./logging.log")  # 输出至文件
        ch = logging.StreamHandler()  # 输出至控制台
        fomatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(fomatter)
        ch.setFormatter(fomatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
        return (logger, fh, ch)  # 把这两个handle都输出，一个区域结束之后断开handle，防止输出数量不断增加


if __name__ == "__main__":
    print("Hello World!")

    yearday = "2002225"     # 8 月 8~10 日对应第 230~232 天，在第 225~232 天的合成产品中
    bands = ["b01", "b02"]  # 只用红和近红外的反射率
    output_file = "../results/Question1.xlsx"  # 输出文件，先写出来再画散点图啥的
    model_params = ["LAI", "LIDFa", "Cab", "Cm", "N"]
    (tts, tto, psi) = (36.2, 0.0, 0.0)  # 经计算，采样地 8 月 9 日上午 10:00 的太阳天顶角是 36.2 度，观测天顶角 == 0，所以相对方位角置 0 即可

    inverter = OyMpInvertor(yearday, output_file, tts, tto, psi, bands=bands, model_params=model_params)
    inverter.run()

    print("Finished.")
