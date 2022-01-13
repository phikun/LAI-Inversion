# LAI-Inversion
- Assignment: Inverting Leaf Area Index (LAI) with MODIS data and ProSAIL model
- 遥感反演作业：用 MODIS 地表反射率产品和 ProSAIL 模型反演叶面积指数
- 问题一：任选一个站点，反演与地面测量数据对应时间的站点周围小区域 LAI，并与基于地面测量数据的 LAI 进行对比分析；
- 问题二：反演站点对应像元任意一年的 LAI，并与 MODIS LAI对比分析

### 数据与方法

#### 站点选取和数据获取

使用网站：http://w3.avignon.inra.fr/valeri/fic_htm/database/main.php 提供的张北的实测 LAI 数据，数据采集时间是 2002 年 8 月 8~10 日，当地经纬度 (114.68°E, 41.27°N)。它公布了高空间分辨率（20m）的有效和真实 LAI 反演结果，将其聚合到 500m 尺度作为真值，与我们利用 MODIS 地表反射率产品反演的进行比较。

MODIS 地表反射率产品和 LAI 产品来自网站：https://modis.ornl.gov/cgi-bin/MODIS/global/subset.pl 处理后的站点附近的 MOD09A1 地表反射率产品和 MOD15A2H LAI 产品。用当期的地表反射率产品做反演，从 2000~2020 年 LAI 产品中获取先验知识。

为将 ProSAIL 模型的模拟结果聚合到 MODIS 传感器上，从网站：https://nwpsaf.eu/downloads/rtcoef_rttov12/ir_srf/rtcoef_eos_1_modis_srf.html 获取搭载在 Terra 卫星上的 MODIS 传感器的光谱响应函数，对 ProSAIL 模型模拟得到的 400~2500nm 各波长的反射率按光谱响应函数加权。

#### 代价函数和优化方法

根据参考文献 [3]，采用便于操作的代价函数形式：

<img src="http://latex.codecogs.com/svg.latex?\mathcal{L}({\rm LAI}, \boldsymbol{x})=\frac{1}{\sigma_{\rm LAI}}\left({\rm LAI}-{\rm LAI}^{\rm priori}\right)^2+\sum_{i=1}^n\left(\frac{1}{\sigma_R_i}\big(R_i-h({\rm LAI}, \boldsymbol{x})\big)^2\right)">

这里 x 是 ProSAIL 模型中除了 LAI 的其他参数，包括平均叶倾角、叶片干物质含量等，可以通过模型的敏感性分析筛选出部分参数，其他参数保持默认值。LAI 的先验和标准差通过除了待反演年份的其他 20 年的 MODIS LAI 产品计算得到。代价函数第二项是各波段的误差求和，这里可以选择 MODIS b01~b07 的部分或全部波段。Ri 表示第 i 个波段的反射率；h 是前向模型，我们这里就用 ProSAIL。

至于各波段的权重 σ_Ri 如何选取，我这里采用网站：https://modis-land.gsfc.nasa.gov/ValStatus.php?ProductID=MOD09# 公布的各波段的不确定度（Uncertainty），尽管我不知道它这个不确定度值是如何计算的，但实验表明代价函数各项的量级相当，实验结果应当合理。

优化方法使用经典的遗传算法，这里采用 scikit-opt 库实现的带缓存加速的遗传算法，没有这个库可以`pip install scikit-opt`装一下。想用遗传算法的原因是，一个别的课（计算方法）有个手写遗传算法的作业，我自以为我写的还行（codes/GA.py）想直接拿过来用；然后发现规则定的不太好，容易快速收敛到局部最优，种群在很长时间内动不了，就找了个现成的包直接用。在 Intel Core i7 上优化一个栅格大概 3 分钟。

#### 计算太阳天顶角

ProSAIL 模型中有个参数是太阳高度角，对于问题一，因为只有一期，在网站：http://www.ab126.com/Geography/1904.html 手动输入经纬度、年月日和时间，它能算出太阳高度角，拿 90° 减去太阳高度角就是太阳天顶角。

对于问题二，一个一个手动计算比较麻烦，然后我编写程序网络交互的能力有限，不会（也可能是懒得）分析这个网站，就用了个简单的计算太阳高度角（天顶角）的方法：经过百度，记 hs 表示太阳高度角，φ, δ, t, day 分别表示纬度、太阳赤纬、时角（正午是 0）、日序，则太阳高度角

<img src="http://latex.codecogs.com/svg.latex?\sin{h_{\rm s}}=\sin\varphi\sin\delta+\cos\varphi\cos\delta\cos t">

其中太阳赤纬的计算公式是：

<img src="http://latex.codecogs.com/svg.latex?\sin\delta=\sin\left(360^\circ\cdot\frac{{\rm day}-80}{365}\right)\sin{23^\circ 26^\prime}">

太阳赤纬基本是模拟太阳从春分点开始，先向北回归线（23°26‘）移动、再向南回归线移动的过程，春分点的日序大概是 80（3 月 21 日，31+28+21=80），所以需要拿实际的日序减去 80 后再算。

### 代码结构

做本实验的代码在 codes/ 文件夹中。

- utility.py 和 converter.py：调用 gdal 库读写 GeoTiff 和 ENVI 标准格式文件，并将栅格转成渔网便于统计。

- prosaillib.py：我不确定咱们 MATLAB 版本的模型能否公开，于是用 MathWorks® MATLAB 自带的打包工具把 ProSAIL 模型打包成了 Python 模块。这个文件负责与 MATLAB 的函数交互。其中`import ProSailPkg`是加载这个打包的模块，需要根据自己打包的模块名称处理一下，并且接口函数及其参数可能都不一样，似乎不太能直接用。
- GA.py：我自己手写的遗传算法，不好用，舍弃。
- optimize.py：优化器，把代价函数最优化的部分单甩出一个 Python 进程，定义 loss_function 类并实现`__call__`方便地修改代价函数形式。
- invertor.py：反演器的基类，实现了反演一个栅格的基本操作（获取 LAI 先验知识、获取各波段的反射率，调用 optimize.py 进行代价函数最优化），其`run`方法是抽象方法，要求子类覆写。
- question1.py 和 question2.py：针对两个问题特殊的反演器，继承了 invertor.py 中定义的反演器基类，向其中添加了自己特殊的方法，并覆写`run`方法实现完整的反演过程。还可以继续继承这两个特殊的优化器，以方便地修改其中某些步骤。

我这里只用了 b01 和 b02 红和近红外波段，用到的 ProSAIL 模型参数是 LAI、ALA（平均叶倾角）、Cab（叶绿素含量）、Cm（叶片干物质含量）和 N（叶片结构参数）。

### References

- [1] PROSPECT 原始论文：Jacquemoud, S., & Baret, F. (1990). PROSPECT: A model of leaf optical properties spectra. *Remote Sensing of Environment*, 34, 75–91.
- [2] SAIL 原始论文：Verhoef,W. (1984). Light scattering by leaf layers with application to canopy reflectance modeling: the SAIL model. *Remote Sensing of Environment*, 16, 125–141.
- [3] 一篇 ProSAIL 综述：Jacquemoud S., Verhoef W., Baret F., Bacour C., & Zarco-Tejada P. J. (2009). PROSPECT+SAIL models: A review of use for vegetation characterization. *Remote Sensing of Environment*, 113, S56–S66.

- [4] MODIS 光谱响应函数下载：https://nwpsaf.eu/downloads/rtcoef_rttov12/ir_srf/rtcoef_eos_1_modis_srf.html

- [5] 各波段不确定性数据来源：https://modis-land.gsfc.nasa.gov/ValStatus.php?ProductID=MOD09#

