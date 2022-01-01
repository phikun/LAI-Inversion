# -*- coding: utf-8 -*-
# LAI Inversion: 用 MODIS 地表反射率数据和 ProSAIL 模型反演叶面积指数
# 常用函数，包括 GeoTiff 类的定义、GeoTiff 的输入输出等

# Author: phikun (201711051122@mail.bnu.edu.cn)
# Date: 2022.01.01

from typing import Union, Tuple, List
from math import isclose
from osgeo import gdal
import numpy as np


# GeoTiff 的元数据类，模仿 MATLAB 简单地 geotiffread、geotiffwrite
class GeotiffInfo:
    def __init__(self, dtype: int, n_rows: int, n_cols: int, trans: Union[List[float], Tuple[float]], proj: str, ndv=None, n_bands: int=1):
        self.dtype = dtype      # 数据类型，GDT_Float32 之类的
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_bands = n_bands  # 波段数，这里常用的基本都是 1
        self.trans = trans      # 坐标角点
        self.proj = proj        # 投影
        self.ndv = ndv          # NoData 值，默认是 None
    
    def __str__(self):
        return f"""GeotiffInfo:
    (n_rows, n_cols) = ({self.n_rows}, {self.n_cols})
    n_bands = {self.n_bands}
      dtype = {self.dtype}
      trans = {self.trans}
       proj = {self.proj}
        ndv = {self.ndv}"""

    def __eq__(self, other):
        """比较两个 GeotiffInfo 是否相同，用于在统计时先确定两个栅格是否能完全重叠"""
        if not isinstance(other, GeotiffInfo):
            raise RuntimeError(f"Value {other} is not a GeoTiffInfo.")
        
        return self.n_rows == other.n_rows and self.n_cols == other.n_cols and self.proj == other.proj and \
               all([isclose(a, b) for (a, b) in zip(self.trans, other.trans)])  # 只要投影、坐标角点、行列号相同就认为两个栅格属性相同


def read_geotiff(fname: str) -> Tuple[np.ndarray, GeotiffInfo]:
    """读入 GeoTiff 栅格，模仿 MATLAB，返回数据矩阵，和栅格基本参数"""
    ds: gdal.Dataset = gdal.Open(fname, gdal.GA_ReadOnly)
    trans = ds.GetGeoTransform()
    proj = ds.GetProjection()

    band: gdal.Band = ds.GetRasterBand(1)
    ndv = band.GetNoDataValue()
    data = band.ReadAsArray()
    dtype = band.DataType
    (n_rows, n_cols) = data.shape
    del ds

    info = GeotiffInfo(dtype, n_rows, n_cols, trans, proj, ndv)
    return (data, info)


def write_geotiff(fname: str, data: np.ndarray, info: GeotiffInfo):
    """写入 GeoTiff 栅格"""
    driver: gdal.Driver = gdal.GetDriverByName("GTiff")
    ds: gdal.Dataset = driver.Create(fname, info.n_cols, info.n_rows, info.n_bands, info.dtype)
    band: gdal.Band = ds.GetRasterBand(1)
    band.WriteArray(data)
    if not info.ndv is None:
        band.SetNoDataValue(info.ndv)

    ds.SetProjection(info.proj)
    ds.SetGeoTransform(info.trans)

    ds.FlushCache()
    del ds
