<div align="center">
<h1>VGGT 网络结构——heads</h1>
</div>

## [HeadAct](./head_act.py)
+ `activate_pose()`：对预测的相机姿态参数进行激活处理，分三个部分 $\mathbf{g}=[\mathbf{q},\mathbf{t},\mathbf{f}]\in\mathbb{R}^9$
    - $\mathbf{t}\in\mathbb{R}^3$：平移向量 $\xrightarrow{\text{激活函数}_1}$ $\mathbf{t}'\in\mathbb{R}^3$
    - $\mathbf{q}\in\mathbb{R}^4$：旋转四元数 $\xrightarrow{\text{激活函数}_2}$ $\mathbf{q}'\in\mathbb{R}^4$
    - $\mathbf{f}\in\mathbb{R}^2$：焦距 $\xrightarrow{\text{激活函数}_3}$ $\mathbf{f}'\in\mathbb{R}^2$
    - 输出 $\mathbf{g}'=[\mathbf{q}',\mathbf{t}',\mathbf{f}']\in\mathbb{R}^9$
+ `activate_head()`：处理 3D 点云预测网络的输出，得到预测结果和置信度
    - 输入：网络输出张量 `output`：${B\times C\times H\times W}\to B\times H\times W\times C$
    - `xyz`：$B\times H\times W\times (C-1)\xrightarrow{\text{激活函数}_1}$ `points`
    - `confidence`：$B\times H\times W\times 1\xrightarrow{\text{激活函数}_2}$ `confidence`
    - 输出：预测结果 `points` $\in\mathbb{R}^{B\times H\times W\times (C-1)}$，置信度 `confidence` $\in\mathbb{R}^{B\times H\times W\times 1}$

## [CameraHead](./camera_head.py)
+ 迭代预测相机姿态参数，得到相机姿态 $\mathbf{g}=[\mathbf{q},\mathbf{t},\mathbf{f}]\in\mathbb{R}^9$
+ 下面所有的 norm 均使用 Layer Normalization
+ `pose_token`：预测得到的相机姿态的 token 表示，维度为 $d$
  - `pose_tokens`：$\mathbb{R}^{B\times S\times d}$ $\xrightarrow{\text{token\_norm()}}$ `pose_tokens`
+ `embed_pose()`：将相机姿态参数编码为 token，本质为 $\xrightarrow{线性层(9\to d)}$
  - `pred_pose`：$\mathbb{R}^9$ $\xrightarrow{\text{embed\_pose()}}$ `module_input`：$\mathbb{R}^d$
+ `poseLN_modulation()`：对 token 进行调制，本质为 $\xrightarrow{\text{SiLU}激活函数}\xrightarrow{线性层(d\to 3d)}$
+ `adaptive_norm()`：自适应层归一化，即只根据输入进行归一化，没有可学习的缩放和偏移参数
  - `module_input`：$\mathbb{R}^d\xrightarrow{\text{poseLN\_modulation()}}$ $\mathbb{R}^{3d}\xrightarrow{分块}$ `gate_msa, scale_msa, shift_msa`：$\mathbb{R}^d$
  - `pose_token_modulated`：$\mathbb{R}^d=$ `gate_msa * (self.adaptive_norm(pose_token) * (1 + scale_msa) + shift_msa) + pose_token`
+ `trunk()`：残差注意力模块，本质为 `trunk_depth` 个 `Block` 模块拼接而成
+ `pose_branch()`：将 token 表示解码为相机姿态，本质为 $\xrightarrow{\text{MLP}(d\to 9)}$
  - `pose_token_modulated`：$\mathbb{R}^d\xrightarrow{\text{trunk()}}$ $\xrightarrow{\text{trunk\_norm()}}$ $\xrightarrow{\text{pose\_branch()}}$ `pred_pose_delta`：$\mathbb{R}^9$
  - `pred_pose`：$\mathbb{R}^9=$ `pred_pose_delta + pred_pose`，初始为 `None` 故直接赋值为 `pred_pose_delta`
+ `activated_pose`：对预测的相机姿态参数进行激活处理
  - `pred_pose`：$\mathbb{R}^9\xrightarrow{\text{activate\_pose()}}$ `activated_pose`：$\mathbb{R}^9$
  - 将每一轮预测的 `activated_pose` 添加到 `pred_pose_list` 中，最后返回 `pred_pose_list`
```mermaid
flowchart LR
    subgraph A [迭代预测]
        A0(pred_pose) --detach+embed_pose--> A1
        A1(module_input) --poseLN_modulation--> A2(shift_msa)
        A1 --poseLN_modulation--> A3(scale_msa)
        A1 --poseLN_modulation--> A4(gate_msa)
        A2 --> A5(pose_token_modulated)
        A3 --> A5
        A4 --> A5
        A6 --adaptive_norm--> A12(pose_token)
        A12 --> A5
        A6 --⊕--> A5
        A5 --trunk+trunk_norm--> A7(pose_token_trunk)
        A7 --pose_branch--> A9(pred_pose_delta)
        A9 --⊕--> A0
        A0 --activate_pose--> A11(activated_pose)
    end
    A_1(empty_pose) --embed_pose--> A1
    A8(pose_token) --token_norm--> A6(pose_token)
    
    linkStyle 0 stroke-width:5px, stroke:green
```

## [DPTHead](./dpt_head.py)
+ 对深度图 $D$、点云图 $P$、跟踪特征 $T$ 和置信度 $c$ 进行预测
+ 下面所有的 norm 均使用 Layer Normalization
+ 根据 `frames_chunk_size` 决定是否分块处理，将 `frames` 送入 `_froward_frame()` 函数
+ `_forward_frame()`：
  - `images`：$B\times S\times C\times H\times W$，共 $S$ 帧，需要截取其中 `frames_start_idx` 到 `frames_end_idx` 的帧
  - `aggregated_tokens_list`：每一层的输出的 token 列表，需要截取 `patch_start_idx` 之后、`frames_start_idx` 到 `frames_end_idx` 的部分
  - `x`：取出 `aggregated_tokens_list` 中第 `layer_idx` 层的 token
  - `projects()`：由卷积层组成，对 `x` 进行卷积得到特征图，`in_features=dim_in`，`out_features=out_channels[i]`，`kernel_size=1`，`stride=1`，`padding=0`
  - `resize_layers()`：对 `x` 进行上采样，得到 `x` 的特征图，包括以下四个部分：
    - `ConvTranspose2d` 层，`in_features=out_channels[0]`，`out_features=out_channels[0]`，`kernel_size=4`，`stride=4`，`padding=0`
    - `ConvTranspose2d` 层，`in_features=out_channels[1]`，`out_features=out_channels[1]`，`kernel_size=2`，`stride=2`，`padding=0`
    - `Identity` 层，不做任何操作
    - `Conv2d` 层，`in_features=out_channels[3]`，`out_features=out_channels[3]`，`kernel_size=3`，`stride=2`，`padding=1`
```mermaid
graph LR
    A(aggregated_tokens_list) --layer_idx--> A1(x)
    A --layer_idx--> A2(x)
    A --layer_idx--> A3(x)
    A --layer_idx--> A4(x)
     A1 --norm--> A1_2(x)
     A1_2 --Conv--> A1_3(x)
     A1_3 --apply_pos_embed*--> A1_4(x)
     A1_4 --ConvTranspose--> A1_5(x)
     A1_5 --> B(out)
     A2 --norm--> A2_2(x)
     A2_2 --Conv--> A2_3(x)
     A2_3 --apply_pos_embed*--> A2_4(x)
     A2_4 --ConvTranspose--> A2_5(x)
     A2_5 --> B(out)
     A3 --norm--> A3_2(x)
     A3_2 --Conv--> A3_3(x)
     A3_3 --apply_pos_embed*--> A3_4(x)
     A3_4 --Identity--> A3_5(x)
     A3_5 --> B(out)
     A4 --norm--> A4_2(x)
     A4_2 --Conv--> A4_3(x)
     A4_3 --apply_pos_embed*--> A4_4(x)
     A4_4 --Conv--> A4_5(x)
     A4_5 --> B(out)
```
+ `_forward_scratch()`：将不同层的特征进行融合
  - `features`：可以分为 `layer1,layer2,layer3,layer4`
  - `layeri_rn()`：对 `layeri` 进行卷积，`in_features=out_channels[i]`，`out_features=features/features*2^(i-1)`，`kernel_size=3`，`stride=1`，`padding=1`
  - `refineneti()`：为 `FeatureFusionBlock` 模块，`features=features`
  - `output_conv1()`：对 `out` 进行卷积得到结果，`in_features=features`，`out_features=features`，`kernel_size=3`，`stride=1`，`padding=1`
```mermaid
graph LR
    A0(features) --> A1(layer1)
    A0 --> A2(layer2)
    A0 --> A3(layer3)
    A0 --> A4(layer4)
    A1 --Conv-->A5(layer1_rn)
    A2 --Conv-->A6(layer2_rn)
    A3 --Conv-->A7(layer3_rn)
    A4 --Conv-->A8(layer4_rn)
    A8 --FeatureFusionBlock-->A10(out)
    A10 --> 1
    A7 --> 1
    1 --FeatureFusionBlock-->A11(out)
    A11 --> l
    A6 --> l
    l --FeatureFusionBlock-->A12(out)
    A12 --> i
    A5 --> i
    i --FeatureFusionBlock-->A13(out)
    A13 --Conv--> A14(out)
    
    style 1 width:0,height:0,opacity:0
    style l width:0,height:0,opacity:0
    style i width:0,height:0,opacity:0
```
+ `forward()`：根据输入的 `aggregated_tokens_list` 预测得到深度图 $D$、点云图 $P$、跟踪特征 $T$ 和置信度 $c$
  - 如果 `frames_chunk_size` 为 `None` 或大于 `S`，则直接完成 `forward()` 操作
  - 否则，将输入按照帧数进行分块，依照 `frames_chunk_size` 的大小依次计算 `frames_start_idx` 和 `frames_end_idx`，分块完成 `forward()` 操作后再将结果拼接起来
  - `output_conv2()`：对 `out` 进行处理得到预测结果，由下面的部分组成：
    - `Conv2d` 层，`in_features=features/2`，`out_features=32`，`kernel_size=3`，`stride=1`，`padding=1`
    - `ReLU` 层
    - `Conv2d` 层，`in_features=32`，`out_features=dim_out`，`kernel_size=1`，`stride=1`，`padding=0`
```mermaid
graph LR
        A0(aggregated_tokens_list) --_forward_frame--> A1(feature)
        A1 --_forward_scratch--> A2(out)
        A2 --custom_interpolate--> A3(out)
        A3 --apply_pos_embed*--> A4(out)
        A4 --feature_only=False--> .
        A4 --feature_only=True--> A6(out)
        . --Conv--> A7(out)
        A7 --activate_head--> A8(points, confidence)
        
        style . width:0,height:0,opacity:0
```

⭐以下是两个组件模块的实现：
### (1). ResidualConvUnit
+ `ResidualConvUnit()`：残差卷积单元
+ 输入输出维度为 `features`$=d$，网络结构如下：
```mermaid
graph LR
        A0(x) --ReLU--> A1(r)
        A1 --Conv--> A2(r)
        A2 --norm*--> A7(r)
        A7 --ReLU--> A3(r)
        A3 --Conv--> A4(r)
        A4 --norm*--> A8(r)
        A8 --> A5(y)
        A0 --⊕--> A5
        
```

### (2). FeatureFusionBlock
+ `FeatureFusionBlock()`：特征融合模块，若 `has_residual` 为 `True`，则将 `x` 与 `x1` 两个特征进行特征融合
+ 输入维度为 `features`$=d_1$，输出维度为 `out_features`$=d_2$，网络结构如下：
```mermaid
graph LR
        A0(x1*) --ResidualConvUnit1*--> A1(r*)
        A2(x) --⊕--> A3(x)
        A1 --> A3
        A3 --ResidualConvUnit2--> A4(x)
        A4 --interpolate--> A5(x)
        A5 --Conv--> A6(y)
```

⭐辅助函数的功能：
### (1). apply_pos_embed
+ `create_uv_grid()`：生成网格坐标 $(u,v)$，将其归一化到 $[-1,1]$ 范围内
+ `position_grid_to_embed()`：将 2D 坐标网格转换为高维的正弦余弦位置嵌入向量
  - `make_sincos_pos_embed()`：将 `pos` 转换为正弦余弦位置嵌入向量
  - 输入 `pos_grid`：$H\times W\times 2\to (HW)\times 2\to$ `x,y`：$\mathbb{R}^{HW}\xrightarrow{\text{make\_sincos\_pos\_embed()}}$`embed_x,embed_y`：$({HW)\times d/2}\to$输出 `embed`：$H\times W\times d$
```mermaid
graph LR
        A1(x: BxCxHxW) --create_uv_grid--> A2(pos_embed: HxWx2)
        A2 --position_grid_to_embed--> A3(pos_embed: HxWxC)
        A3 --expand--> A4(pos_embed: BxCxHxW)
        A4 --> A5(x: BxCxHxW)
        A1 --⊕--> A5
```
### (2). custom_interpolate
+ `custom_interpolate()`：自定义插值函数，对 `x` 进行插值操作
  - 本质仍为 `F.interpolate()`，但能够解决输入元素数量超过 `INT_MAX` 的问题，将其进行分块后分别处理
