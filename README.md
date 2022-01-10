# LAI-Inversion
- Assignment: Inverting Leaf Area Index (LAI) with MODIS data and ProSAIL model
- 遥感反演作业：用 MODIS 地表反射率产品和 ProSAIL 模型反演叶面积指数

### 太阳高度角的计算公式

记 $h_{\rm s}$ 表示太阳高度角，$\varphi,\delta,t$ 分别表示纬度、太阳赤纬、时角（正午是 0），则太阳高度角
$$
\sin{h_{\rm s}}=\sin\varphi\sin\delta+\cos\varphi\cos\delta\cos t
$$
其中太阳赤纬的计算公式是：
$$
\sin\delta=\sin\left(360^\circ\cdot\frac{t}{365}\right)\sin{23^\circ 26^\prime}
$$

### References

- MODIS 光谱响应函数下载：https://nwpsaf.eu/downloads/rtcoef_rttov12/ir_srf/rtcoef_eos_1_modis_srf.html

- 各波段不确定性数据来源：https://modis-land.gsfc.nasa.gov/ValStatus.php?ProductID=MOD09#

