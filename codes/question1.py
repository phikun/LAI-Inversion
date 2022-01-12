# -*- coding: utf-8 -*-
# LAI Inversion: 用 MODIS 地表反射率数据和 ProSAIL 模型反演叶面积指数
# 作业一：一期、多个像元各自反演 LAI，并与 SPOT 高分辨率合成到 500m 分辨率的参考值比较

# Author: phikun (201711051122@mail.bnu.edu.cn)
# Date: 2022.01.12

from typing import Collection, Tuple, Dict
from overrides import overrides
from tqdm import tqdm
from glob import glob
import geopandas as gpd
import pandas as pd

from converter import raster2fishnet
from invertor import invertor
import utility as util


class OyMpInvertor(invertor):
    def __init__(self, year: int, daynum: int, output_file: str, tts: float, tto: float, psi: float, 
                       indices=None, bands: Collection[str]=None, model_params: Collection[str]=None, spot_band_id: int=1):
        """
        构造函数
        :param year:          待反演的年份
        :param daynum:        待反演的日序
        :param output_file:   结果文件，干脆把每个像元反演得到的 LAI、SPOT 向上聚合的 LAI 和 MODIS 同期 LAI 产品的值写到文件里
        :param tts, tto, psi: 太阳天顶角、观测天顶角、相对方位角，直接在 ProSAIL 模型中指定
        :param indices:       待反演像元的索引，若为 None 则指定是该区域内所有完全落入 SPOT 影像范围内的栅格
        :param bands:         反演用到的波段，若为 None 则表示使用全部 b01~b07 共 7 个波段
        :param model_params:  要用到的 ProSAIL 模型参数，若为 None 则使用所有参数
        :param spot band_id:  用于比较的 SPOT 结果的波段号
        """
        log_file = "./logging/OneTime-MultiPixel-logging.log"
        middle_result_file = "./logging/OneTime-MultiPixel-middle-results.txt"
        super(OyMpInvertor, self).__init__(bands=bands, model_params=model_params, log_fname=log_file, middle_res=middle_result_file)

        (self._year, self._day) = (year, daynum)
        self._output_file = output_file
        self._obs_geom = {"tts": tts, "tto": tto, "psi": psi}  # 观测几何

        self._spot_band_id = spot_band_id
        self._spot_file = "../data/ZhangBei/SPOTZhangbei20020809TOA_VarBioMaps/SPOTZhangbei20020809TOA_VarBioMaps.bil"  # SPOT 反演结果文件

        if indices is not None:  # 指定待反演栅格的行列号
            self._indices = indices
        else:
            rows = [r for r in range(7, 13) for _ in range(29, 37)]
            cols = [c for _ in range(7, 13) for c in range(29, 37)]
            self._indices = list(zip(rows, cols))

    def __get_ref_lai(self) -> Dict[Tuple[int, int], float]:
        """用 SPOT 的结果合成参考值，"""
        # Step1: 用 self._indice 索引，获取需要的 MODIS 栅格，
        modis_lai_raster = glob(f"{self._lai_path}*{self._year}{self._day:03d}*Lai_500m.tif")[0]
        modis_gdf = raster2fishnet(modis_lai_raster)
        rc = list(zip(modis_gdf["Row"], modis_gdf["Column"]))
        modis_gdf.index = rc                         # 用每个像元的行列号作为索引
        modis_gdf = modis_gdf.loc[self._indices, :]  # 用self.__indices 从索引中取值，这里关心渔网的空间位置

        # Step2: 获取指定波段编号的 SPOT 栅格，并投影到与 MODIS 的坐标系一致
        spot_gdf = raster2fishnet(self._spot_file, raster_band_id=self._spot_band_id).to_crs(modis_gdf.crs)
        spot_gdf["Value"] = spot_gdf["Value"].astype(float) / 1000.0  # SPOT 反演产品的DN值除以 1000 才是需要的叶面积指数
        
        # Step3: 将 MODIS 和 SPOT 栅格叠加，统计完全包含在某个 MODIS 像元中的 SPOT LAI 的均值
        crossover = gpd.sjoin(modis_gdf, spot_gdf, how="left", op="contains")   # contains 做空间叠加
        means = crossover.reset_index().groupby("index")["Value_right"].mean()  # 计算落入某个 MODIS 像元的所有 SPOT 像元的均值
        ref_lai = dict(means)
        return ref_lai        

    def __get_modis_lai(self) -> Dict[Tuple[int, int], float]:
        """获取当期 MODIS LAI 产品各像元的值，用于与反演结果比较"""
        fname = glob(f"{self._lai_path}*{self._year}{self._day:03d}*Lai_500m.tif")[0]
        (data, _) = util.read_geotiff(fname)
        values = [data[row, col] / 10.0 for (row, col) in self._indices]
        res = dict(zip(self._indices, values))
        return res

    @overrides
    def run(self):
        # Step0: 用 SPOT 的结果合成 LAI 参考值
        ref_lai = self.__get_ref_lai()

        # Step1: 当期 MODIS LAI 产品的像元值
        modis_lai = self.__get_modis_lai()

        # Step2: 逐项元反演
        inv_lai = [self._base_invert_one_pixcel(self._year, self._day, row, col) for (row, col) in tqdm(self._indices)]

        df = pd.DataFrame({"Inversion": inv_lai, "SPOT": list(ref_lai.values()), "MODIS": list(modis_lai.values())}, index=self._indices)
        df.to_excel(self._output_file, engine="openpyxl", index=True)


if __name__ == "__main__":
    print("Hello World!")

    (year, day) = (2002, 225)     # 8 月 8~10 日对应第 230~232 天，在第 225~232 天的合成产品中
    bands = ["b01", "b02"]        # 只用红和近红外的反射率
    output_file = "../results/Question1-trueLai.xlsx"  # 输出文件，先写出来再画散点图啥的
    model_params = ["LAI", "LIDFa", "Cab", "Cm", "N"]
    (tts, tto, psi) = (36.2, 0.0, 0.0)  # 经计算，采样地 8 月 9 日上午 10:00 的太阳天顶角是 36.2 度，观测天顶角 == 0，所以相对方位角置 0 即可

    inverter = OyMpInvertor(year, day, output_file, tts, tto, psi, bands=bands, model_params=model_params, spot_band_id=2)  # 用真实 LAI
    inverter.run()

    print("Finished.")
