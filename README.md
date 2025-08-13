# UAVCalibration: 基于反投影与卫星图像匹配的航拍定位方案

## Requirements

-   [uv](https://docs.astral.sh/uv/getting-started/installation/): optional but highly recommend
-   [Git](https://git-scm.com/downloads): optional
-   [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) ~12.6: required for GPU inference
-   [UAV-VisLoc Dataset](https://github.com/IntelliSensing/UAV-VisLoc): required for demo script

## Installation and demo

### Download

Clone repository

```bash
# clone repository with submodules
git clone --recurse-submodules https://github.com/carseny/UAVCalibration.git && cd UAVCalibration
```

Download UAV_VisLoc_example or other datasets and place them into `./datasets`

### Set up environment with uv (recommend)

Change the pytorch source to your own cuda version in `./pyproject.toml` at `[[tool.uv.index]].url`. e.g. `https://download.pytorch.org/whl/cpu` for pytorch with CPU inference.

```toml
# Custom source for PyTorch with CUDA 12.6
[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cu126"
explicit = true
```

Create virtual environment and install this repo with all requirements.

```bash
uv venv
uv pip install -e .
```

Run demonstration scripts in `./scripts`

```bash
uv run ./scripts/demo.py
```

### Set up environment without uv

Create virtual environment and install this repo with all requirements. (pytorch cpu version)

```bash
python -m venv .venv
./.venv/scripts/activate
pip install -e .
```

(Optional) Delete torch cpu version and install pytorch with your cuda version.
```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Run demonstration scripts in `./scripts`

```bash
./.venv/scripts/activate
python ./scripts/demo.py
```

## Scripts

### Interactive Demonstration Using UAV_VisLoc_example Dataset

`./scripts/demo.py`

Moving mouse pointer on one image and get corresponding point on the other.

Control keys:

-   Esc: Exit
-   A: Previous image
-   W: 10th Previous image
-   D: Next image
-   S: 10th Next image

![Interactive Demonstration](./assets/demo_UAV-VisLoc.jpg)

## Compute Flow Chart

```mermaid
flowchart
    raw_image --> raw_hw

    focal_length --> cam_mat
    raw_hw --> cam_mat

    yaw --> rot_mat
    pitch --> rot_mat
    roll --> rot_mat

    cam_mat --> rect_trans
    rot_mat --> rect_trans

    lonlat --> rect_crs_trans
    height --> rect_crs_trans
    cam_mat --> rect_crs_trans
    rot_mat --> rect_crs_trans

    rect_crs_trans --> get_sat{get_sat}
    raw_hw --> get_sat
    get_sat --> sat_image
    get_sat --> sat_crs_trans

    rect_trans --> rect_image
    raw_image --> rect_image

    rect_image --> match_trans
    sat_image --> match_trans

    sat_crs_trans --> crs_trans["crs_trans
    (img_xy -> crs_coord)"]
    match_trans --> crs_trans
```

# 相关的研究

## 数据集

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

## 坐标计算

### 坐标系统

#### WGS84 (World Geodetic System)

-   **经度 (λ)：** 角度值（-180° 到 +180°），以本初子午线为基准，向东为正（东经），向西为负（西经）。
-   **纬度 (φ)：** 角度值（-90° 到 +90°），以赤道为基准，向北为正（北纬），向南为负（南纬）。
-   **大地高 (h)：** 从 WGS84 椭球面沿法线方向向上的距离（单位：米）。高于椭球面为正，低于为负。

#### UTM (通用横轴墨卡托坐标系统)

它与全球纬度和经度不同之处在于，它将地球分为 60 个区域，并将每个区域投影到平面上作为其坐标的基准。

可以获得局部地区最小失真的图像

#### GeoTIFF 图像坐标系

-   这是一个 **离散的、以像素为单位的坐标系** 。
-   原点 `(0, 0)` 通常位于 **图像的左上角** 。
-   `i` 轴（列索引）向右（东）方向递增。
-   `j` 轴（行索引）向下（南）方向递增。
-   坐标表示为 `(column, row)` 或 `(x_pixel, y_pixel)`，例如 `(100, 200)` 表示第 200 行、第 100 列（注意行号通常从上到下增加）。

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

设三维空间中的点为(x,y,z)，二维图像平面上的点为(x',y')，则它们之间的映射关系可以表示为：

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
-   **俯仰角$\theta$（Pitch）**：围绕 Y 轴（左右）旋转的角度
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

### 问题描述

需要建立从无人机图像到卫星图像的精确（<10m）对应关系，从而获得无人机图像上目标的精确坐标信息。

在经过反投影矫正后，仍面临以下问题

-   无人机 6 DOF 数据不精确导致的偏移/尺度/旋转不对应
-   拍摄角度变化导致的视角、遮挡关系不对应
-   拍摄时间变化导致的风格不对应
-   常见的图像匹配任务专注于匹配图像内某一部分，而航拍图像与卫星图像视野极广

### Cross-view Geo-localization 论文调研

#### Cross-view geo-localization: a survey

-   背景与动机
    -   随着大规模地理标注数据集和机器学习技术的发展，跨视角地理定位成为研究热点
    -   在自动驾驶、增强现实、环境监测等领域具有重要应用
-   本文贡献
    -   全面梳理特征工程与深度学习两大类主流方法
    -   总结数据集、评价指标与现有挑战
    -   展望未来研究方向与应用前景

##### 问题定义

-   输入：未知坐标的地面视角图像
-   数据库：带有已知经纬度的航拍/卫星图像集合
-   输出：查询图像的最可能地理坐标

##### 方法演变

-   基于像素/几何的早期方法

    -   像素级地理对齐（Geodetic alignment）
    -   传感器模型+正射校正

-   基于手工特征的方法

    -   特征提取+匹配：SIFT、SURF、词汇树等
    -   同视图与跨视图匹配策略

-   基于深度学习的方法

    -   Siamese 网络与三元组损失
    -   Capsule 网络
    -   GAN 生成与合成式检索
    -   Transformer 与自注意力机制

##### 多视角匹配中的主要挑战

-   视角差异与尺度变化
-   光照变化、遮挡与动态物体
-   数据规模大，检索效率瓶颈

##### 数据集与评价指标

-   常用数据集
    -   Pittsburgh 250k、CVUSA、CVACT 等
-   评价指标
    -   Top‑k 精度、召回率、定位误差

##### 方法对比与性能分析

-   不同方法在各数据集上的性能比较
-   特征工程 vs. 深度学习的优缺点

##### 未来展望与应用

-   融合多源、多模态信息
-   轻量化模型与在线检索加速
-   在智能交通、应急响应、增强现实等场景中的落地

#### CLIP

[CLIP](https://github.com/openai/CLIP)使用了对比学习方法，在大量互联网上的图像-文本对进行训练，能够从语义上对 zero-shot 图像进行特征提取与匹配。

#### Pix2Map

[Pix2Map](https://arxiv.org/pdf/2301.04224) 针对驾驶场景，从车辆的第一视角图像中直接推断出对应街区的地图拓扑结构，以根据需要不断更新和扩展现有地图。。

![Pix2Map Architecture](https://pix2map.github.io/figures/3.png)

Pix2Map 使用 ResNet 与 Transformer 模型分别将图像特征与地图特征进行编码到特征向量，使用类似 CLIP 的对比学习机制进行特征对齐，从而实现图像到地图的匹配。

优点：

-   使用向量相似度度量图像与地图之间的相似度，计算相对高效。
-   拥有类似 CLIP 的 zero-shot 学习能力，可以处理未见过的场景。

缺点：

-   需要大量的标注数据进行训练。
-   相对于传统方法来说，模型可解释性差、鲁棒性可能不足。

#### Multiple-environment Self-adaptive Network for Aerial-view Geo-localization

使用一个额外的风格预测网络，解决无人机图像在不同环境下（雨天、雾天等）识别准确性问题。

#### Sample4Geo: Hard Negative Sampling For Cross-View Geo-Localisation

![Sample4Geo](https://ar5iv.labs.arxiv.org/html/2303.11851/assets/x2.png)

##### Symmetric InfoNCE Loss

$ℒ(q,R)_\text{InfoNCE} = -log \frac{exp(q ⋅ r_+ / τ)}{ \sum_{i=0}^{N} exp(q ⋅ r_i / τ)}$

> $q$ denotes an encoded street-view, the so-called query, and $R$ is a set of encoded satellite images called references. Only one positive $r_i$, namely $r_+$ matches to $q$. The InfoNCE loss uses the dot-product to calculate the similarity between query and reference images and is low when the query and the positive match are similar, and high when the negative $r_i$ are dissimilar to $q$. As loss function for the similarity between the views the cross-entropy is calculated. The temperature parameter $τ$ is a hyperparameter that can either be learned [30] or set to a static value.

> So far, InfoNCE loss has mostly been used in a non-symmetric fashion for unsupervised representation learning for images [28, 29]. A symmetric formulation showed to be useful in multi-modal pre-training [30] to bridge the gap between the modalities. Therefore we utilise this loss function in the same symmetric fashion to leverage the flow of information in both directions: satellite-view to street-view and vice versa. In the InfoNCE loss, a positive example is always contrasted with N-1 negative examples, where N denotes the batch size, thus delimiting many examples at once. **But in cases where there are several positive examples, such as University-1652, this requires a custom sampler to prevent multiple positives for the same ground truth label in one batch.** We provide an ablation study of the importance of symmetry in the InfoNCE loss and a comparison to the triplet loss in our supplementary material.

##### Model Architecture

使用 Siamese 网络，用一个权重共享的 ConvNeXt 作为两个视图的单一编码器

#### Each Part Matters: Local Patterns Facilitate Cross-view Geo-localization

![LPN](https://ar5iv.labs.arxiv.org/html/2008.11646/assets/x3.png)

认为图像不同环形分区具有不同语义，不同图片相同分区内语义一一对应。（个人感觉有点过于针对无人机环绕拍摄的数据集了）

提出 Local Pattern Network (LPN)，在全局池化前分离不同环形区域并分别池化。

#### A Transformer-Based Feature Segmentation and Region Alignment Method for UAV-View Geo-Localization

-   使用预训练的分类网络按 patch 进行特征向量提取
-   将特征向量所有维度取平均得到 thermal value 标量（似乎有点蠢）
-   通过 thermal value 将图像分割为不同区域
-   将每类的分类结果分别进行平均池化和对齐

### Image Matching 论文调研

#### Image Matching from Handcrafted to Deep Features: A Survey

##### Image matching 的主要目标（任务类型）

1. 图像配准（Image Registration）与全景拼接（Stitching）
   将两张或多张图像对齐、融合为一个一致图像，以生成全景或做时序比较。
2. 结构恢复与视觉定位（3D Reconstruction / Visual Localization / SLAM）
   通过图像间对应关系恢复三维结构或估计相机姿态，用于自动导航、地图构建等。
3. 视觉归巢（Visual Homing）
   机器人仅凭视觉信息确定当前位置相对于“家”所在方向，基于匹配特征或 motion flow 估计归向矢量。
4. 图像检索 / 目标识别
   通过匹配图像特征实现物体识别、实例检索或匹配同一目标的不同拍摄图像。

##### 方法上的分类（方法类别）

1. 区域／模板匹配（Area‑based 或 Template matching）

-   直接基于图像灰度／像素块相似度（如交叉相关或 SAD）滑动匹配。
-   适合小尺度、刚性、重叠多的图像，但对旋转、尺度、遮挡、光照变化不鲁棒。

2. 基于特征的匹配（Feature‑based Matching）

    1. 特征检测（Feature Detection）

        - 手工设计（handcrafted）：Harris、DoG／LoG、MSER、FAST、Hessian-Affine 等经典算子。
        - 学习型（learning-based）：通过深度网络训练检测更加稳健的关键点或局部兴趣点。

    2. 特征描述（Feature Description）

        - 手工描述子：如 SIFT（128‑维）、SURF、ORB（FAST+BRIEF）等，针对旋转、尺度变化设计的鲁棒描述子。
        - 学习型描述子：如使用 CNN、Siamese 网络、Patch‑based 方法，或 end‑to‑end 联合学习 detector+descriptor。

    3. 特征匹配与几何筛除

        - 利用 brute‑force 或 KNN、ratio test 等方法比较描述子；再通过 RANSAC 或几何模型估计剔除错误匹配。

3. 深度学习端到端方法（Detector‑free 或 中端／末端学习匹配）

-   Detector‑free 模型：直接从输入图像对中学习密集或半稠密对应关系（如 LoFTR、SuperGlue 等 Transformer 或 CNN‑based 方法）。
-   联合模块（end‑to‑end matcher）：整合检测、描述、匹配、几何估计为统一可学习模块，用于姿态估计、单阶段重建等。

#### SIFT（Scale-Invariant Feature Transform）

SIFT 是一种经典的图像局部特征提取算法，由 David Lowe 在 1999 年提出并在 2004 年完善。它能够在图像中检测并描述对尺度缩放、旋转、亮度变化甚至一定程度的视角变化和仿射变换保持稳定的关键特征点。

#### SuperPoint

用于检测兴趣点和编码描述符。

-   使用生成的数据监督预训练
-   在无标注数据集上进一步自监督训练

##### 网络架构

![SuperPoint Architecture](https://ar5iv.labs.arxiv.org/html/1712.07629/assets/x3.png)

1. 网络
   使用同一个卷积 (VGG Net) 编码兴趣点和描述符，并分别通过 PixelShuffle 和 Bi-Cubic 插值上采样到原始尺寸。
2. 损失函数

    - 兴趣点：交叉熵
    - 描述符：使用余弦相似度，最小化相邻对应位置的距离、最大化其他距离，使用了加权项 $λ_d$ 帮助平衡由于负对应关系多于正对应关系而产生的不平衡

##### 训练策略

1. 使用合成图像预训练

    - 使用实时渲染的二维图形（灰度）以生成兴趣点数据

2. 仿射适应

    - 使用预训练模型在无标注数据集 (MS-COCO) （灰度）上生成伪标签
    - 基于透视变化不变性，同一图像的同一位置应具有相同兴趣点和描述符。
      先应用透视变换，再将结果变化回来，并取所有变换后伪标签的平均值作为下一次迭代的伪标签。（相当于数据增强了）

#### SuperGlue

基于注意力图神经网络的特征点匹配中端，从两组给定特征点与描述符寻找匹配点。

-   需要使用特征点检测前端，与 SuperPoint 相性很好
-   监督训练
-   全程梯度可传导，可以端到端训练。但是只能训练描述符网络，冻结特征点检测部分

##### 网络架构

![SuperGlue Architecture](https://ar5iv.labs.arxiv.org/html/1911.11763/assets/x4.png)

-   使用 self/cross attention 进行图像间信息提取
-   使用 sinkhorn 算法在余弦相似度上寻找最佳匹配，梯度可传导

#### LoFTR

仿照 SuperPoint + SuperGlue，但是抛弃特征点检测部分，在密集的粗粒度特征图像上进行特征点匹配，最后在细粒度上进行精校。

-   监督训练

##### 网络架构

![LoFTR Architecture](https://ar5iv.labs.arxiv.org/html/2104.00680/assets/x2.png)

-   使用 CNN 与 U-Net 结构提取图像特征
-   使用 self/cross attention 进行图像间信息提取，其中使用线性的注意力算法

#### Lightglue: Local feature matching at light speed

SuperGlue 的跟进研究，主要优化了性能，精度提升较小

![lightglue Architecture](https://ar5iv.labs.arxiv.org/html/2306.13643/assets/x4.png)

##### Adaptive depth and width

-   **Early Exit**
-   **Point pruning**: 随着 LightGlue 聚合上下文，它可以早期发现某些点是不可匹配的，并因此将它们从后续层中排除。其他非重复点在后期层中被排除。这减少了推理时间和搜索空间，最终快速找到好的匹配。

##### 与 SuperGlue 对比

-   **位置编码**：SuperGlue 使用 MLP 编码绝对点位置，并在早期与描述子融合，模型在整个层中容易忘记这种位置信息。相反，LightGlue 依赖于一种相对编码，这种编码在不同图像之间更容易比较，并在每个自注意力单元中添加。这使得更容易利用位置信息并提高了深层层的准确性。
-   **预测头**：SuperGlue 通过使用 Sinkhorn 算法解决可微最优传输问题来预测分配。它包括许多行和列归一化的迭代，这在计算和内存方面都很昂贵。SuperGlue 添加了一个垃圾桶来拒绝不可匹配的点。我们发现垃圾桶将所有点的相似度得分纠缠在一起，从而导致次优的训练动态。LightGlue 将相似性和可匹配性分开，这些更易于预测。这也产生了更干净的梯度。
-   **深度监督**：由于 Sinkhorn 的计算成本高昂，SuperGlue 无法在每一层后进行预测，只能在最后一层进行监督。LightGlue 较轻的头部使其能够在每一层进行分配预测并进行监督，这加快了收敛速度，并允许在任何一层后退出推理，这是 LightGlue 效率提升的关键。

#### 算法比较

![benchmark](https://ar5iv.labs.arxiv.org/html/2306.13643/assets/x1.png)

**\*LightGlue matches sparse features faster and better** than existing approaches like SuperGlue. Its adaptive stopping mechanism gives a fine-grained control over the speed vs. accuracy trade-off. Our final, optimized model ⋆ delivers an accuracy closer to the dense matcher LoFTR at an 8× higher speed, here in typical outdoor conditions.\*
