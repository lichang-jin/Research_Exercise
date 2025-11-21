<div align="center">
<h1>VGGT 网络结构——heads</h1>
</div>

## [HeadAct](./head_act.py)
+ `activate_pose()`：对预测的相机姿态参数进行激活处理，分三个部分 $\mathbf{g}=[\mathbf{q},\mathbf{t},\mathbf{f}]\in\mathbb{R}^9$
    - $\mathbf{t}\in\mathbb{R}^3$：平移向量 $\xrightarrow{\text{激活函数}_1}$ $\mathbf{t}'\in\mathbb{R}^3$
    - $\mathbf{q}\in\mathbb{R}^4$：旋转四元数 $\xrightarrow{\text{激活函数}_2}$ $\mathbf{q}'\in\mathbb{R}^4$
    - $\mathbf{f}\in\mathbb{R}^2$：焦距 $\xrightarrow{\text{激活函数}_3}$ $\mathbf{f}'\in\mathbb{R}^2$
    - $\mathbf{g}'=[\mathbf{q}',\mathbf{t}',\mathbf{f}']\in\mathbb{R}^9$
+ `activate_head()`：处理 3D 点云预测网络的输出，得到 3D 点坐标和置信度
    - 输入：网络输出张量 `output`：${B\times C\times H\times W}\to B\times H\times W\times C$
    - `xyz`：$B\times H\times W\times (C-1)\xrightarrow{\text{激活函数}_1}$ `points`
    - `confidence`：$B\times H\times W\times 1\xrightarrow{\text{激活函数}_2}$ `confidence`
    - 输出：3D 点坐标 $\mathbf{p}\in\mathbb{R}^{B\times H\times W\times 3}$，置信度 $\mathbf{c}\in\mathbb{R}^{B\times H\times W\times 1}$

## [CameraHead](./camera_head.py)