# -*- coding: utf-8 -*-
# LAI Inversion: 用 MODIS 地表反射率数据和 ProSAIL 模型反演叶面积指数
# 作业二：反演站点对应像元任意一年的LAI，并与 MODIS LAI 对比分析

# Author: phikun (201711051122@mail.bnu.edu.cn)
# Date: 2022.01.12

from typing import Collection, Tuple, List, Dict
from math import pi, cos, sin, asin
from overrides import overrides
from glob import glob
from tqdm import tqdm
import pandas as pd
import json
import os

from invertor import invertor
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


class OpWyInvertor(invertor):
    def __init__(self, year: int, output_file: str, lat: float, time_theta: float, index: Tuple[int, int], 
                       bands: Collection[str]=None, model_params: Collection[str]=None):
        """
        构造函数
        :param year:         待反演的年份
        :param output_file:  结果文件，把反演结果和同期 MODIS LAI 产品写到文件里 
        :param lat:          像元纬度，十进制度，北纬为正，用于计算太阳天顶角
        :param time_theta:   时角，十进制度，正午是 0，上午为负、下午为正，用于计算太阳天顶角
        :param index:        待反演像元的索引
        :param bands:        反演用到的波段，若为 None 则表示使用全部 b01~b07 共 7 个波段
        :param model_params: 要用到的 ProSAIL 模型参数，若为 None 则使用所有参数
        """
        log_file = "./OnePixcel-WholeYear-logging.log"
        middle_result_file = "./OnePixcel-WholeYear-middle-results.txt"
        super(OpWyInvertor, self).__init__(bands=bands, model_params=model_params, log_fname=log_file, middle_res=middle_result_file)

        self._lat = lat
        self._year = year
        self._time_theta = time_theta
        self._index = index
        self._output_file = output_file
        (self._obs_geom["tto"], self._obs_geom["psi"]) = (0.0, 0.0)  # 天底观测，不需要相对方位角

    def __get_modis_lai(self) -> Dict[int, float]:
        """获取一年内每一天的 MODIS LAI 产品的值"""
        daynums = list(range(1, 365, 8))  # 日序，八天合成
        lais: List[float] = []

        for daynum in daynums:
            fname = glob(f"{self._lai_path}*{self._year}{daynum:03d}*Lai_500m.tif")[0]
            (data, _) = util.read_geotiff(fname)
            lai = data[self._index[0], self._index[1]] / 10.0  # MODIS LAI 产品的尺度银子是 10
            lais.append(lai)
        
        return dict(zip(daynums, lais))

    def _invert_one_pixcel(self, year: int, daynum: int, row: int, col: int) -> float:
        """反演一个栅格的值，调用基类的 base 方法，在调用前计算太阳天顶角，并赋给 self.obs_geom 的 tts 属性"""
        tts = calc_tts(daynum, self._lat, self._time_theta, degree=True)
        self._obs_geom["tts"] = tts  # 计算太阳高度角

        lai = self._base_invert_one_pixcel(year, daynum, row, col)
        return lai

    @overrides
    def run(self):
        # Step0: 当期 MODIS LAI 产品的像元值
        modis_lai = self.__get_modis_lai()

        # Step1: 逐日反演
        inv_lai = [self._invert_one_pixcel(self._year, daynum, self._index[0], self._index[1]) for daynum in tqdm(range(1, 365, 8))]

        df = pd.DataFrame({"Inversion": inv_lai, "MODIS": list(modis_lai.values())}, index=range(1, 365, 8))
        df.to_excel(self._output_file, engine="openpyxl", index=True)


# 在 question2 的反演器中添加细节：以先验 LAI 的 μ±3σ 作为 LAI 的取值范围
class OpWyInvertor_withLaiPriority(OpWyInvertor):
    def __init__(self, year: int, output_file: str, lat: float, time_theta: float, index: Tuple[int, int], 
                       bands: Collection[str]=None, model_params: Collection[str]=None):
        """构造函数，与其父类一样"""
        super(OpWyInvertor_withLaiPriority, self).__init__(year, output_file, lat, time_theta, index, bands=bands, model_params=model_params)

    @overrides
    def _invert_one_pixcel(self, year: int, daynum: int, row: int, col: int) -> float:
        """反演一个栅格的值，算出太阳天顶角，并根据 LAI 先验信息修改 LAI 的取值范围"""
        tts = calc_tts(daynum, self._lat, self._time_theta, degree=True)
        self._obs_geom["tts"] = tts

        (lai_mean, lai_std) = self._get_lai_mean_and_std(year, daynum, row, col)
        opt_info_file = "../data/optimize_information.json"  # 模型参数文件，修改其中 LAI 的取值范围
        with open(opt_info_file, "r") as fin:
            opt_info = json.load(fin)
            fin.seek(0)              # 把文件指针拨到最前面
            lines = fin.readlines()  # 保留原本的文件
        
        opt_info["ModelParams"]["LAI"]["default"] = lai_mean
        opt_info["ModelParams"]["LAI"]["lb"] = max(0.0, lai_mean - 3.0 * lai_std)  # μ-3σ 作为 LAI 下界
        opt_info["ModelParams"]["LAI"]["ub"] = min(5.0, lai_mean + 3.0 * lai_std)  # μ+3σ 作为 LAI 上界
        with open(opt_info_file, "w") as fout:
            json.dump(opt_info, fout)  # 把修改后的模型参数写入文件

        lai = self._base_invert_one_pixcel(year, daynum, row, col)
        with open(opt_info_file, "w") as fout:
            fout.writelines(lines)  # 原封不动写回原来的模型参数

        return lai


if __name__ == "__main__":
    print("Hello World!")

    year = 2020
    index = (10, 32)  # 采样点所在栅格的行列号
    (lat, time) = (41.278889, -30.0)  # 纬度、时角（卫星过境时间大概是上午十点，所以时角是 -30）
    output_file = "../results/Question2-withLaiPriority.xlsx"
    bands = ["b01", "b02"]
    model_params = ["LAI", "LIDFa", "Cab", "Cm", "N"]

    inv = OpWyInvertor_withLaiPriority(year, output_file, lat, time, index, bands=bands, model_params=model_params)
    inv.run()

    print("Finished.")
