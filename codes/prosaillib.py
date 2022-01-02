# -*- coding: utf-8 -*-
# A Python export of ProSail model: 
#   Basic definition of <prosaillib> class

# Author: __phi__ (201711051122@mail.bnu.edu.cn)
# Date: 2021.11.11

from typing import Union, Tuple, List
from scipy.interpolate import interp1d
import numpy as np
import os


class pyprosail:
    def __init__(self, path: str=None, filter_funcion: bool=True):
        """
        :param path: environment path where MATLAB runtime <mclmcrrt9_5.dll> locate.
        :param filter_function: 是否需要用 MODIS 的光谱响应函数校正，默认需要
        """
        self.__n_bands = 7                                          # 只考虑 MODIS 的前 7 个波段 
        self.__central_wl = [645, 858, 469, 555, 1240, 1640, 2130]  # MODIS 前 7 个波段的中心波长
        self.__filter_function = self.__get_filter_params()         # 获取前 7 个波段的光谱响应函数
        self.__filter_flag = filter_funcion                         # 是否要用光谱响应函数

        if not path is None:  # Add environment path
            os.environ["PATH"] += f";{path};"

        import ProSailPkg
        
        self.__mtdll = ProSailPkg.initialize()
    
    def __del__(self):
        self.__mtdll.terminate()

    def run(self, N: float=1.5, Cab: float=50.0, Car: float=8.0, Cbrown: float=0.0, Cw: float=0.015, Cm: float=0.005, 
                      LIDFa: float=57.0, LIDFb: float=0.0, TypeLidf: int=2,
                      LAI: float=3.5, hspot: float=0.2, tts: float=30.0, tto: float=0.0, psi: float=0.0, psoil: float=1.0):
        """
        Main function of ProSail model, and return reflectance of each wavelength
        :param N:      in PROSEPCT, structure coefficient, number of leaf layers
        :param Cab:    in PROSPECT, chlorophyll content (μg.cm-2)
        :param Car:    in PROSPECT, carotenoid content (μg.cm-2)
        :param Cbrown: in PROSPECT, brown pigment content (arbitrary units)
        :param Cw:     in PROSPECT, water, EWT (cm)
        :param Cm:     in PROSPECT, leaf dry matter, LMA (g.cm-2)
        :param LIDFa:    in SAIL, leaf, controls the average leaf slope
        :param LIDFb:    in SAIL, leaf, controls the distribution's bimodality
        :param TypeLidf: in SAIL, leaf, decides which type of distribution
        :param LAI:      in SAIL, leaf area index
        :param hspot:    in SAIL, hot spot parameter
        :param tts:      in SAIL, solar zenith angle (°)
        :param tto:      in SAIL, observer zenith angle (°)
        :param psi:      in SAIL, relatively azimuth (°)
        :param psoil:    in SAIL, ratio of day soil
        :return: 返回 MODIS 第 1~7 波段各波段的反射率
        """
        assert TypeLidf in {1, 2}  # There are only 2 options of typeLidf

        # Call MATLAB package
        tmp = self.__mtdll.prosail_5B(N, Cab, Car, Cbrown, Cw, Cm, LIDFa, LIDFb, TypeLidf, LAI, hspot, tts, tto, psi, psoil)
        ref = np.zeros(2501)                      # 0~2500 共 2501 个波长
        ref[400:] = np.array(tmp).squeeze()       # 把最初 400 个波长空出来，方便用 MODIS 的光谱响应函数合成

        if self.__filter_flag is False:           # 不要光谱响应函数，返回中心波长的反射率
            return ref[self.__central_wl]
        else:                                     # 要光谱响应函数，按光谱响应函数加权后返回 
            return [np.sum(ref * self.__filter_function[i]) for i in range(self.__n_bands)]

    @staticmethod
    def __get_filter_param_one_band(fname: str) -> np.ndarray:
        """计算一个波段的光谱响应函数参数：从文件中读入以波数表示的光谱响应函数，转成纳米波长，并对响应值做归一化"""
        with open(fname, "r") as fin:
            [fin.readline() for _ in range(4)]  # 空出前 4 行
            lines = list(map(str.strip, fin.readlines()))
        wavenumbers = np.array([float(line.split()[0]) for line in lines])
        filter_values = np.array([float(line.split()[1]) for line in lines])
        wavelengths = 1E7 / wavenumbers  # 把波数转成纳米波长，它原本波数的范围是 cm^-1，所以直接拿 1E7 去除就行了
        interp = interp1d(wavelengths, filter_values, kind="linear")  # 给定插值结点，用线性就可以了

        (left_, right_) = (np.ceil(wavelengths[-1]), np.floor(wavelengths[0]))  # 插值点的左右端点
        xx = np.arange(left_, right_ + 1)
        yy = interp(xx)
        values = yy / yy.sum()  # 光谱响应值归一化
        
        res = np.zeros(2501)
        res[int(left_):int(right_ + 1)] = values
        return res

    def __get_filter_params(self) -> List[np.ndarray]:
        """获取所有 7 个波段的光谱响应函数"""
        path = "../data/Terra filter functiions"
        fnames = [f"{path}/rtcoef_eos_1_modis_srf_ch{i:02d}.txt" for i in range(1, self.__n_bands + 1)]
        filter_params = [pyprosail.__get_filter_param_one_band(fname) for fname in fnames]
        return filter_params


if __name__ == "__main__":
    print("Hello World!")

    # ============================== 【测试带光谱响应函数和不带光谱响应函数的版本】 ==============================
    # path = "E:/MATLAB/Program/runtime/win64"
    # prosail1 = pyprosail(path=path, filter_funcion=False)  # 先测试不用波谱响应函数的
    # res1 = prosail1.run()

    # prosail2 = pyprosail(path=path, filter_funcion=True)   # 再测试要光谱响应函数的
    # res2 = prosail2.run()

    # print(res1)
    # print(res2)

    print("Finished.")
