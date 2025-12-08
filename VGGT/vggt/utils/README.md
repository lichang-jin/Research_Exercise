<div align="center">
<h1>辅助函数</h1>
</div>

## [Quaternion & Matrix](./rotation.py)
+ 旋转四元数与旋转矩阵之间的转换
### 1. quaternion_to_matrix
+ `quaternion`：$\mathbb{R}^4=[i,j,k,r]$
+ `matrix`：$\mathbb{R}^{3\times3}$
  - $s=2{||\mathbf{q}||_2^{-2}}$
  - $\mathbf{M}=\begin{bmatrix}1-s(j^2+k^2) & s(ij-kr) & s(ik+jr)\\s(ij+kr) & 1-s(i^2+k^2) & s(jk-ir)\\s(ik-jr) & s(jk+ir) & 1-s(i^2+j^2)\end{bmatrix}$
### 2. matrix_to_quaternion
+ `matrix`：$\mathbb{R}^{3\times3}=\begin{bmatrix}m_{00} & m_{01} & m_{02}\\m_{10} & m_{11} & m_{12}\\m_{20} & m_{21} & m_{22}\end{bmatrix}$
+ `q_sqrt`：$[1+m_{00}+m_{11}+m_{22}, 1+m_{00}-m_{11}-m_{22}, 1-m_{00}+m_{11}-m_{22}, 1-m_{00}-m_{11}+m_{22}]=2s[r^2, i^2, j^2, k^2]$
+ `q_abs`：$\sqrt{2s}[|r|, |i|, |j|, |k|]$
+ `q_candidates`：分别假设不同的分量最大，得到四个候选四元数
    - $[1+m_{00}+m_{11}+m_{22}, m_{21}-m_{12}, m_{02}-m_{20}, m_{10}-m_{01}]=2s[r^2,ri,rj,rk]=2sr\mathbf{q}$
    - $[m_{21}-m_{12}, 1+m_{00}-m_{11}-m_{22}, m_{10}+m_{01}, m_{02}+m_{20}]=2si\mathbf{q}$
    - $[m_{02}-m_{20}, m_{10}+m_{01}, 1-m_{00}+m_{11}-m_{22}, m_{21}+m_{12}]=2sj\mathbf{q}$
    - $[m_{10}-m_{01}, m_{02}+m_{20}, m_{21}+m_{12}, 1-m_{00}-m_{11}+m_{22}]=2sk\mathbf{q}$
+ 选择最大的分量作为除数求出四元数（最稳定）

## [Distortion](./distortion.py)
+ 相机畸变矫正

### 1. apply_distortion
+ 输入归一化后的坐标 $(u,v)$
+ 距离计算：$r=u^2+v^2$
+ 畸变参数：$[k_1,k_2,p_1,p_2]$
+ 畸变参数个数为 1 时，只包含径向畸变，去畸变函数为 $u = u+uk_1r^2$
+ 畸变参数个数为 2 时，只包含径向畸变，去畸变函数为 $u = u+u(k_1r^2+k_2r^4)$
+ 畸变参数个数为 4 时，包含径向畸变和切向畸变，去畸变函数为 
  - $u = u+u(k_1r^2+k_2r^4)+2p_1uv+p_2(r^2+2u^2)$
  - $v=v+v(k_1r^2+k_2r^4)+2p_2uv+p_1(r^2+2v^2)$

### 2. signal_undistortion
+ 对相机坐标系下的点坐标进行非迭代去畸变处理

### 3. iterative_undistortion
+ 对相机坐标系下的点坐标进行迭代去畸变处理
+ 使用牛顿法迭代，记 $f(u,v)=$ `apply_distortion()`，则对于第 $t$ 轮迭代：
  - $[u_{c,t},v_{c,t}]=f(u_t,v_t)$
  - $[du_t,dv_t]=[u_0,v_0]-[u_{c,t},v_{c,t}]$
  - 雅可比矩阵 $J=\left[\begin{matrix}\frac{\partial f_u}{\partial u} & \frac{\partial f_u}{\partial v}\\\frac{\partial f_v}{\partial u} & \frac{\partial f_v}{\partial v}\end{matrix}\right]$
  - 解方程 $J[\Delta u_t, \Delta v_t]=[du_t,dv_t]$
  - $[u_{t+1},v_{t+1}]=[u_t,v_t]+[\Delta u_t, \Delta v_t]$
  - 若 $\||[\Delta u_t, \Delta v_t]||<\epsilon$，则停止迭代

## [Geometry](./geometry.py)
+ 多视几何相关函数，涉及到相机参数、相机坐标系、世界坐标系、图像坐标系之间的变换
### 1. unproject_depth_map_to_point_map
+ 将深度图反投影到三维点云
+ 内参矩阵：$K=\left[\begin{matrix}f_x&&u_0\\&f_y&v_0\\&&1\end{matrix}\right]$
+ 外参矩阵：增广矩阵 $[R\ \ \ \ t]$，其中 $R\in\mathbb{R}^{3\times 3}$，$t\in\mathbb{R}^3$
  - 由 $\frac{f}{Z_c}=\frac{u-u_0}{X_c}=\frac{v-v_0}{Y_c}$，得到在每个 $(h,w)$ 处的相机坐标系下的三维点坐标 $(X_c,Y_c,Z_c)$
  - 由 $\left[\begin{matrix}X_c\\Y_c\\Z_c\end{matrix}\right]=[R\ \ \ \ t]\left[\begin{matrix}X_w\\Y_w\\Z_w\\1\end{matrix}\right]$，得到在世界坐标系下的三维点坐标 $(X_w,Y_w,Z_w)$

### 2. image_points_from_camera_points
+ 将相机坐标系下的三维点投影到图像坐标系
+ `camera_points`：相机坐标系下的三维点 $\xrightarrow{归一化(/Z)}\xrightarrow{畸变矫正}$ `camera_coords` 
+ 最后乘以内参矩阵 $K$，得到图像坐标系下的二维点

### 3. camera_points_from_image_points
+ 将追踪的图像坐标系下的二维点反投影到相机坐标系
+ $X_c'=\frac{X_c}{Z_c}=\frac{u-u_0}{f_x}$，$Y_c'=\frac{Y_c}{Z_c}=\frac{v-v_0}{f_y}$
+ $(X_c', Y_c')\xrightarrow{畸变矫正}(X_c'', Y_c'')$

### 4. project_world_points_to_camera
+ 将世界坐标系下的三维点投影到相机坐标系
+ $\left[\begin{matrix}X_c\\Y_c\\Z_c\end{matrix}\right]=[R\ \ \ \ t]\left[\begin{matrix}X_w\\Y_w\\Z_w\\1\end{matrix}\right]$

## [Pose Encoding & Camera Extrinsic / Intrinsic](./pose_encoding.py)
+ 相机位姿编码与相机外参、内参之间的转换

### 1. camera_param_to_pose_encoding
+ 通过相机外参、内参得到相机位姿编码
+ 外参矩阵 $[R\ \ \ \ t]\to \mathbf{t}\in\mathbb{R}^3$，$[R\ \ \ \ t]\to \mathbf{R}\xrightarrow{\text{matrix\_to\_quaternion}}\mathbf{q}\in\mathbb{R}^4$
+ 内参矩阵 $K=\left[\begin{matrix}f_x&&u_0\\&f_y&v_0\\&&1\end{matrix}\right]\to\left[\begin{matrix}f_x\\f_y\end{matrix}\right]\to\mathbf{f}=\left[\begin{matrix}2\arctan(\frac{H}{2f_y})\\2\arctan(\frac{W}{2f_x})\end{matrix}\right]\in\mathbb{R}^2$
+ 位姿编码 $\mathbf{p}=[\mathbf{t},\mathbf{q},\mathbf{f}]\in\mathbb{R}^9$

### 2. pose_encoding_to_camera_param
+ 通过相机位姿编码得到相机外参、内参
+ $\mathbf{p}=[\mathbf{t},\mathbf{q},\mathbf{f}]\to\mathbf{t}\in\mathbb{R}^3$，$\mathbf{q}\xrightarrow{\text{quaternion\_to\_matrix}}\mathbf{R}\in\mathbb{R}^{3\times3}$，得到外参矩阵 $[R\ \ \ \ t]$
+ $\mathbf{f}\to\left[\begin{matrix}f_x\\f_y\end{matrix}\right]\to K=\left[\begin{matrix}f_x&&\frac{W}{2}\\&f_y&\frac{H}{2}\\&&1\end{matrix}\right]$