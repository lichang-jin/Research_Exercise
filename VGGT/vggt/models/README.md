<div align="center">
<h1>VGGT 网络结构——models</h1>
</div>

## [Aggregator](./aggregator.py)
+ VGGT 的核心模块，负责将图像和相机参数编码为 token，并使用交替注意力机制进行特征提取。
+ `silice_expand_and_flatten()`：将输入的 `camera_token`：$1\times 2\times 1\times C$ 和 `register_token`：$1\times 2\times 4\times C$ 扩展到维度 $B\times S\times 1/4 \times C$。
    - 第一帧 $S=1$ 用 $[:, 0, :, :]$ 扩展，其他 $S-1$ 帧用 $[:, 1, :, :]$ 扩展，以区分第一帧（作为基准坐标系）。

```mermaid
flowchart LR
    C(camera_token)--slice_expand_and_flatten-->D(camera_token)
    E(register_token)--slice_expand_and_flatten-->F(register_token)
    A(images)--DINOv2_ViT-->B(patch_token)
    D --<1>--> G(tokens)
    F --<4>--> G(tokens)
    B --> G(tokens)
    L(camera_pos)--0-->M(camera_pos)
    J(register_pos)--0-->K(register_pos)
    H(patch_pos)--PositionGetter-->I(patch_pos)
    M --> O(pos)
    K --> O(pos)
    I --> O(pos)
    G-->AA
    O-->AA
    subgraph AA [Alternating-Attention]
    Q--Global-Attention-->P
    P(tokens)--Frame-Attention--> Q(tokens)
    end
    AA --> R(out)
```

## [VGGT](./vggt.py)
+ VGGT 的主体模块，输入图像和待追踪点，输出相机参数、重建结果和追踪结果
```mermaid
flowchart LR
    
    subgraph 1[VGGT]
    D1(aggregated_tokens_list)--CameraHead-->E(pose_enc_list)
    D2(images, aggregated_tokens_list)--DPTHead-->F(depth, depth_confidence)
    D2--DPTHead-->G(points, points_confidence)
    D3(images, aggregated_tokens_list, query_points)--TrackHead-->H(track_list, visibility, track_confidence)
    end
    A(images)--Aggregator-->B(aggregated_tokens_list)
    A --> 1
    B --> 1
    C(query_points) --> 1
```