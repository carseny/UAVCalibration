## 已有的研究

### 数据集

1. [UAV-VisLoc：无人机视觉定位的大规模数据集](https://github.com/IntelliSensing/UAV-VisLoc)

    - 数据包含日期时间、经纬度、高度以及三轴信息，以及拍摄区域的卫星图
    - 俯仰、滚转几乎都是 0（即垂直俯视拍摄）
    - 未提供镜头详细参数，无法计算出每个像素对应的精确位置
    - 有[issue](https://github.com/IntelliSensing/UAV-VisLoc/issues/5)反应经纬度似乎不准

2. [SUES-200](https://github.com/Reza-Zhu/SUES-200-Benchmark)

    - 数据包含针对同一个目标的卫星图和无人机拍摄的不同角度的图像，不包含详细的位置信息。

3. [DenseUAV](https://github.com/Dmmm1997/DenseUAV)

    - 提出了一种基于视觉的无人机自定位方案，可以比对无人机拍摄图像与卫星图像中的特征来估计无人机的位置。
    - 数据集没公开需要申请

## 反投影

### 相机内参矩阵

将三维相机坐标系中的点投影到二维图像平面。

$$
K=
\begin{bmatrix}
f_x & s & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{bmatrix}
$$

设三维空间中的点为(x,y,z)，二维图像平面上的点为(x',y')（归一化到[0,1])，则它们之间的映射关系可以表示为：

$$
\begin{bmatrix}
x' \\
y' \\
1
\end{bmatrix}
= K \cdot \begin{bmatrix}
x/z \\
y/z \\
1
\end{bmatrix}
$$

| 参数       | 符号  | 物理意义                    | 单位 | 典型值示例 |
| ---------- | ----- | --------------------------- | ---- | ---------- |
| x 轴焦距   | $f_x$ | 相机 x 轴方向焦距长度       | 像素 | 3000 px    |
| y 轴焦距   | $f_y$ | 相机 y 轴方向焦距长度       | 像素 | 3000 px    |
| 主点坐标 x | $c_x$ | 光轴与图像平面的交点 x 坐标 | 像素 | 1920 px    |
| 主点坐标 y | $c_y$ | 光轴与图像平面的交点 y 坐标 | 像素 | 1080 px    |
| 倾斜系数   | $s$   | 图像坐标轴的倾斜程度        | -    | 0          |

#### 焦距：($f_x, f_y$) （像素）

表示从相机光心到图像平面的距离（所以其实可以理解成像距）

非对称焦距 ($f_x != f_y$)： 由像素非正方形引起（常见于手机相机）

#### 主点坐标：($c_x, c_y$) （像素）

主光轴与图像平面交点的坐标，理想情况下位于图像中心

#### 倾斜系数 ($s$)

现代相机：多数$s≈0$（传感器与光轴垂直）

### 姿态角与旋转矩阵

<!-- ![欧拉角](https://upload.wikimedia.org/wikipedia/commons/a/a1/Eulerangles.svg)
*三个欧拉角： (α, β, γ)。蓝色的轴是 xyz-轴，红色的轴是 XYZ-坐标轴。绿色的线是交点线 (N)。* -->

![姿态角](https://i-blog.csdnimg.cn/blog_migrate/c54b7445c7d483d9d8d65356b540f48c.png)
_这个图似乎画错了，一般是右手系，z 应向下_

-   **偏航角$\psi$（Yaw）**：围绕 Z 轴（上下）旋转的角度
-   **俯仰角$(\theta)$（Pitch）**：围绕 Y 轴（左右）旋转的角度
-   **翻滚角$\phi$（Roll）**：围绕 X 轴（前后）旋转的角度

$$
R_z=\begin{bmatrix}
cos(\psi) & -sin(\psi) & 0 \\
sin(\psi) & cos(\psi) & 0 \\
0 & 0 & 1
\end{bmatrix}
$$

$$
R_y=\begin{bmatrix}
cos(\theta) & 0 & sin(\theta) \\
0 & 1 & 0 \\
-sin(\theta) & 0 & cos(\theta)
\end{bmatrix}
$$

$$
R_x=\begin{bmatrix}
1 & 0 & 0 \\
0 & cos(\phi) & -sin(\phi) \\
0 & sin(\phi) & cos(\phi)
\end{bmatrix}
$$

根据[参考资料](https://blog.csdn.net/qq_45518988/article/details/120338303)，一般按照 Z-Y-X 顺序旋转，因此从飞行器坐标换算到世界坐标应该依次按 X-Y-Z 顺序应用旋转矩阵：

$$
R = R_z · R_y · R_x
$$

### OpenCV warpPerspective 函数

warpPerspective 使用的 3×3 单应矩阵（Homography Matrix） 表示两个平面之间的透视变换关系，其物理意义可分解如下（设矩阵为 H）：

$$
H = \begin{bmatrix}
h_{11} & h_{12} & h_{13} \\
h_{21} & h_{22} & h_{23} \\
h_{31} & h_{32} & h_{33}
\end{bmatrix}
$$

设源平面上的点为(x,y)，目标平面上的点为(x′,y′)，则它们之间的映射关系可以用齐次坐标表示为：

$$
\begin{bmatrix}
x'_{hom} \\
y'_{hom} \\
w'
\end{bmatrix}
= H \cdot \begin{bmatrix}
x \\
y \\
1
\end{bmatrix}
$$

$$
x' = \frac{x'_{hom}}{w'} \\
y' = \frac{y'_{hom}}{w'}
$$

## 图像匹配

### 论文调研

#### 传统图像特征提取算法

-   SIFT（Scale-Invariant Feature Transform，尺度不变特征变换）是一种经典的图像局部特征提取算法，由 David Lowe 在 1999 年提出并在 2004 年完善。它能够在图像中检测并描述对尺度缩放、旋转、亮度变化甚至一定程度的视角变化和仿射变换保持稳定的关键特征点。
-   HOG（方向梯度直方图）特征提取是一种基于图像局部梯度方向统计分布的目标描述方法，在行人检测等任务中展现出对几何形变和局部光照变化的较强鲁棒性。

#### CLIP

[CLIP](https://github.com/openai/CLIP)使用了对比学习方法，在大量互联网上的图像-文本对进行训练，能够从语义上对 zero-shot 图像进行特征提取与匹配。

#### Pix2Map

[Pix2Map](https://arxiv.org/pdf/2301.04224) 针对驾驶场景，从车辆的第一视角图像中直接推断出对应街区的地图拓扑结构，以根据需要不断更新和扩展现有地图。。

![](https://pix2map.github.io/figures/3.png)

Pix2Map 使用 ResNet 与 Transformer 模型分别将图像特征与地图特征进行编码到特征向量，使用类似 CLIP 的对比学习机制进行特征对齐，从而实现图像到地图的匹配。

优点：

-   使用向量相似度度量图像与地图之间的相似度，计算相对高效。
-   拥有类似 CLIP 的 zero-shot 学习能力，可以处理未见过的场景。

缺点：

-   需要大量的标注数据进行训练。
-   相对于传统方法来说，模型可解释性差、鲁棒性可能不足。

#### TODO

-   Lending Orientation to Neural Networks for Cross-View Geo-Localization (2019)
-   Each Part Matters: Local Patterns Facilitate Cross-View Geo-Localization (2020)
-   University-1652: A Multi-view Multi-source Benchmark for Drone-based Geo-localization (2020, dataset)
-   UAV-Satellite View Synthesis for Cross-View Geo-Localization (2022)
-   Multiple-environment Self-adaptive Network for Aerial-view Geo-localization (2022)

### 卫星图处理

UAV-VisLoc 数据集中，卫星图的像素比例尺似乎是与经纬度相对应的，这导致卫星图像看起来像是被压扁了，需要在纵向坐标时除以 cos(lat) 来纠正这种畸变。
