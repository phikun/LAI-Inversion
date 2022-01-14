# -*- coding: utf-8 -*-
# LAI Inversion: 用 MODIS 地表反射率数据和 ProSAIL 模型反演叶面积指数
# 作业二：反演站点对应像元任意一年的LAI，并与 MODIS LAI 对比分析
# 2021.01.14 更新：取消观测几何参数，因为可以直接从 MODIS 产品中读取；这已添加到基类的 _base_invert_one_pixcel 方法中

# Author: phikun (201711051122@mail.bnu.edu.cn)
# Date: 2022.01.12

from typing import Collection, Tuple, List, Dict
from overrides import overrides
from glob import glob
from tqdm import tqdm
import pandas as pd
import json
import os

from invertor import invertor
import utility as util


class OpWyInvertor(invertor):
    def __init__(self, year: int, output_file: str, index: Tuple[int, int], 
                       bands: Collection[str]=None, model_params: Collection[str]=None):
        """
        构造函数
        :param year:         待反演的年份
        :param output_file:  结果文件，把反演结果和同期 MODIS LAI 产品写到文件里 
        :param index:        待反演像元的索引
        :param bands:        反演用到的波段，若为 None 则表示使用全部 b01~b07 共 7 个波段
        :param model_params: 要用到的 ProSAIL 模型参数，若为 None 则使用所有参数
        """
        log_file = "./logging/OnePixcel-WholeYear-logging.log"
        middle_result_file = "./logging/OnePixcel-WholeYear-middle-results.txt"
        super(OpWyInvertor, self).__init__(bands=bands, model_params=model_params, log_fname=log_file, middle_res=middle_result_file)

        self._year = year
        self._index = index
        self._output_file = output_file

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
        lai = self._base_invert_one_pixcel(year, daynum, row, col)  # 早期版本要在这里计算太阳天顶角，实际上完全不需要
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
    def __init__(self, year: int, output_file: str, index: Tuple[int, int], 
                       bands: Collection[str]=None, model_params: Collection[str]=None):
        """构造函数，与其父类一样"""
        super(OpWyInvertor_withLaiPriority, self).__init__(year, output_file, index, bands=bands, model_params=model_params)

    @overrides
    def _invert_one_pixcel(self, year: int, daynum: int, row: int, col: int) -> float:
        """反演一个栅格的值，算出太阳天顶角，并根据 LAI 先验信息修改 LAI 的取值范围"""
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

        lai = self._base_invert_one_pixcel(year, daynum, row, col)  # 在这里面已获取了观测几何参数
        with open(opt_info_file, "w") as fout:
            fout.writelines(lines)  # 原封不动写回原来的模型参数

        return lai


if __name__ == "__main__":
    print("Hello World!")

    year = 2020
    index = (10, 32)  # 采样点所在栅格的行列号
    output_file = "../results/Question2-withLaiPriority-rightAngle.xlsx"
    bands = ["b01", "b02"]
    model_params = ["LAI", "LIDFa", "Cab", "Cm", "N", "psoil"]

    inv = OpWyInvertor_withLaiPriority(year, output_file, index, bands=bands, model_params=model_params)
    inv.run()

    print("Finished.")
