# -*- coding: utf-8 -*-
# LAI Inversion: 用 MODIS 地表反射率数据和 ProSAIL 模型反演叶面积指数
# 数据转换器：包括栅格转渔网、投影转换等，用于将 MODIS 数据与 SPOT 数据叠加

# Author: phikun (201711051122@mail.bnu.edu.cn)
# Date: 2022.01.01

from typing import List
from shapely.geometry import Point, Polygon
import geopandas as gpd
import pandas as pd
import numpy as np

import utility as util


def raster2fishnet(raster_file: str) -> gpd.GeoDataFrame:
    """栅格转渔网，每个 Polygon 的属性是该像元各波段的值，并在属性表中添加各像元的行列号；忽略 NoData!"""
    (data, info) = util.read_geotiff(raster_file)

    # Step1: 生成渔网的角点坐标
    (n_rows, n_cols) = data.shape
    (x0, dxx, dxy, y0, dyx, dyy) = info.trans  # 6 个仿射变换参数
    cols = np.array(range(0, n_cols + 1))
    rows = np.array(range(0, n_rows + 1))
    (xx, yy) = np.meshgrid(cols, rows)
    proj_x = x0 + dxx * xx + dxy * yy
    proj_y = y0 + dyx * xx + dyy * yy

    # Step2: 按行优先的顺序遍历栅格，生成 Polygon，且属性值即是像素值
    polygons: List[Polygon] = []
    values = []
    (row_indices, col_indices) = ([], [])
    for i in range(n_rows):
        for j in range(n_cols):
            if data[i, j] == info.ndv:  # 忽略 NoDataValue
                continue
            pnts = [Point(proj_x[i][j],         proj_y[i][j]), 
                    Point(proj_x[i][j + 1],     proj_y[i][j + 1]), 
                    Point(proj_x[i + 1][j + 1], proj_y[i + 1][j + 1]), 
                    Point(proj_x[i + 1][j],     proj_y[i + 1][j]), 
                    Point(proj_x[i][j],         proj_y[i][j])]
            polygon = Polygon(pnts)
            polygons.append(polygon)
            values.append(data[i, j])
            row_indices.append(i)
            col_indices.append(j)

    # Step3: 生成 GeoDataFrame 并返回
    gdf = gpd.GeoDataFrame(pd.DataFrame({"Value": values, "Row": row_indices, "Column": col_indices}), geometry=polygons, crs=info.proj)
    return gdf


if __name__ == "__main__":
    print("Hello World!")

    # ==================== 【测试 raster2fishnet 函数】 ====================
    # raster_file = "../data/MODIS LAI/MOD15A2H.A2002233.h26v04.006.2015150032221_Lai_500m.tif"
    # gdf = raster2fishnet(raster_file)
    # gdf.to_file("../test/raster2fishnet/MOD15A2H.A2002233_LAI_500m.shp")
    # raster_file = "../data/ZhangBei/SPOTZhangbei20020809TOA_VarBioMaps/SPOTZhangbei20020809TOA_VarBioMaps.bil"  # 此时有效叶面积指数恰好是第 1 个波段！
    # gdf = raster2fishnet(raster_file)
    # gdf["Value"] = gdf["Value"].astype(float) / 1000.0  # 除以 1000 后才是真正的 LAI
    # gdf.to_file("../test/raster2fishnet/SPOTZhangbei20020809TOA_LAIeff_20m.shp")

    print("Finished.")
