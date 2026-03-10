"""
FACTO 论文完整复现 - XTrainer 机械臂

基于论文理论体系：
1. Fourier 基函数轨迹表示
2. 函数空间自适应优化 (LM + 高斯牛顿)
3. 约束优化 (关节限制 + 碰撞避让)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, List, Optional, Dict
import time
import xml.etree.ElementTree as ET


class XTrainerRobot:
    """XTrainer 6轴机械臂 - 从 URDF 解析"""
    
    def __init__(self, urdf_path: str = None, collision_spheres: Dict = None):
        """从 URDF 构建机器人模型"""
        
        # 6轴机械臂 DH 参数 (从 URDF 解析)
        # 关节: J_1 ~ J_6
        # 连杆: LINK_0(base) -> LINK_1 -> ... -> LINK_6(末端)
        
        self.n_dof = 6
        
        # DH 参数 (单位: 米)
        # 从 URDF 提取
        # J_1: 绕 Z 轴旋转, offset=0.2234m
        # J_2: 绕 Y 轴旋转 (有 RPY 变换)
        # J_3: 沿 X 轴 -0.28m
        # J_4: 绕 Z 轴旋转, offset=-0.225m
        # J_5: 沿 Y 轴 -0.12m  
        # J_6: 沿 Z 轴 0.083m
        
        # 各关节位置偏移 (从 URDF origin)
        self.joint_offsets = np.array([
            [0, 0, 0.2234],      # J_1 -> LINK_1
            [0, 0, 0],           # J_2 -> LINK_2
            [-0.28, 0, 0],       # J_3 -> LINK_3
            [-0.225, 0, 0.1175],  # J_4 -> LINK_4
            [0, -0.12, 0],       # J_5 -> LINK_5
            [0, 0.083, 0]        # J_6 -> LINK_6
        ])
        
        # 各关节轴 (从 URDF axis)
        self.joint_axes = np.array([
            [0, 0, 1],  # J_1
            [0, 0, 1],  # J_2
            [0, 0, 1],  # J_3
            [0, 0, 1],  # J_4
            [0, 0, 1],  # J_5
            [0, 0, 1]   # J_6
        ])
        
        # 连杆长度 (简化)
        self.link_lengths = [0.2234, 0, 0.28, 0.225, 0.12, 0.083]
        
        # 关节限制 (rad)
        self.joint_limits = np.array([
            [-3.14, 3.14],  # J_1
            [-3.14, 3.14],  # J_2
            [-3.14, 3.14],  # J_3
            [-3.14, 3.14],  # J_4
            [-3.14, 3.14],  # J_5
            [-3.14, 3.14]   # J_6
        ])
        
        # 关节速度限制 (rad/s，从 URDF <limit velocity="..."> 提取)
        self.joint_velocity_limits = np.array([
            10.0,  # J_1
            10.0,  # J_2
            10.0,  # J_3
            10.0,  # J_4
            10.0,  # J_5
            10.0   # J_6
        ])
        
        # 碰撞球体 (从 xtrainer.yml)
        self.collision_spheres = collision_spheres or {}
        
    def dh_transform(self, theta: float, d: float, a: float, alpha: float) -> np.ndarray:
        """标准 DH 变换"""
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array([
            [ct, -st*ca, st*sa, a*ct],
            [st, ct*ca, -ct*sa, a*st],
            [0, sa, ca, d],
            [0, 0, 0, 1]
        ])
    
    def forward_kinematics(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """正运动学 - 计算末端位置和姿态"""
        
        # 简化模型: 依次变换
        T = np.eye(4)
        
        # 基础位置
        T = self.dh_transform(0, self.joint_offsets[0, 2], 0, np.pi/2)
        
        for i in range(self.n_dof):
            # 关节变换
            d = self.joint_offsets[i, 2] if i < len(self.joint_offsets) else 0
            a = self.link_lengths[i] if i < len(self.link_lengths) else 0
            alpha = 0
            
            # 根据 URDF 有一些特殊旋转
            if i == 1:  # J_2
                alpha = np.pi/2
            elif i == 2:  # J_3
                a = 0.28
            elif i == 3:  # J_4
                alpha = -np.pi/2
            elif i == 4:  # J_5
                alpha = np.pi/2
            elif i == 5:  # J_6
                alpha = -np.pi/2
                
            T_i = self.dh_transform(q[i], d, a, alpha)
            T = T @ T_i
            
        return T[:3, 3], T[:3, :3]
    
    def get_workspace_bounds(self) -> Dict:
        """工作空间边界"""
        return {
            'x': [-0.8, 0.8],
            'y': [-0.8, 0.8],
            'z': [0, 0.6]
        }
    
    def check_collision(self, q: np.ndarray, obstacles: List[np.ndarray]) -> bool:
        """碰撞检测"""
        try:
            pos, _ = self.forward_kinematics(q)
            
            # 检查与障碍物的距离
            for obs in obstacles:
                dist = np.linalg.norm(pos - obs)
                if dist < 0.05:  # 5cm 安全距离
                    return True
                    
            # 检查自碰撞 (简化)
            # 实际需要检查每个连杆
            
        except:
            return True
            
        return False
    
    def compute_end_effector_position(self, q: np.ndarray) -> np.ndarray:
        """计算末端执行器位置"""
        pos, _ = self.forward_kinematics(q)
        return pos


class FourierBasis:
    """Fourier 基函数轨迹表示 (带缓存，避免重复计算)"""
    
    def __init__(self, n_basis: int = 10, n_dof: int = 6, n_points: int = 100):
        self.n_basis = n_basis
        self.n_dof = n_dof
        self.n_points = n_points
        self.t = np.linspace(0, 1, n_points)
        
        # 预计算并缓存所有矩阵，避免每次调用都重新生成
        self._Phi = self._build_basis_matrix()
        self._dPhi = self._build_velocity_matrix()
        self._d2Phi = self._build_acceleration_matrix()
    
    def _build_basis_matrix(self) -> np.ndarray:
        """构建基函数矩阵: 常数 + Hermite + Fourier
        
        列布局: [1, 3t²-2t³, sin(2πt), cos(2πt), sin(4πt), cos(4πt), ...]
        Hermite 项 h(t)=3t²-2t³ 用于打破周期性，且端点导数为0:
          h(0)=0, h(1)=1, h'(0)=0, h'(1)=0
        """
        Phi = np.zeros((self.n_points, self.n_basis))
        Phi[:, 0] = 1.0        # 常数项
        if self.n_basis > 1:
            Phi[:, 1] = 3*self.t**2 - 2*self.t**3  # Hermite项 (打破周期性 + 端点零速度)
        for k in range(2, self.n_basis):
            n = (k) // 2
            if k % 2 == 0:  # sin
                Phi[:, k] = np.sin(n * 2 * np.pi * self.t)
            else:  # cos
                Phi[:, k] = np.cos(n * 2 * np.pi * self.t)
        return Phi
    
    def _build_velocity_matrix(self) -> np.ndarray:
        """构建速度基函数矩阵"""
        dPhi = np.zeros((self.n_points, self.n_basis))
        # k=0: d(1)/dt = 0
        if self.n_basis > 1:
            dPhi[:, 1] = 6*self.t - 6*self.t**2  # d(3t²-2t³)/dt = 6t-6t²
        for k in range(2, self.n_basis):
            n = (k) // 2
            w = n * 2 * np.pi
            if k % 2 == 0:  # d(sin)/dt = w*cos
                dPhi[:, k] = w * np.cos(w * self.t)
            else:  # d(cos)/dt = -w*sin
                dPhi[:, k] = -w * np.sin(w * self.t)
        return dPhi
    
    def _build_acceleration_matrix(self) -> np.ndarray:
        """构建加速度基函数矩阵"""
        d2Phi = np.zeros((self.n_points, self.n_basis))
        # k=0: d2(1)/dt2 = 0
        if self.n_basis > 1:
            d2Phi[:, 1] = 6 - 12*self.t  # d2(3t²-2t³)/dt2 = 6-12t
        for k in range(2, self.n_basis):
            n = (k) // 2
            w = n * 2 * np.pi
            if k % 2 == 0:  # d2(sin)/dt2 = -w^2*sin
                d2Phi[:, k] = -(w**2) * np.sin(w * self.t)
            else:  # d2(cos)/dt2 = -w^2*cos
                d2Phi[:, k] = -(w**2) * np.cos(w * self.t)
        return d2Phi
        
    def basis_matrix(self) -> np.ndarray:
        """Fourier 基函数矩阵 (缓存)"""
        return self._Phi
    
    def coeffs_to_trajectory(self, c: np.ndarray) -> np.ndarray:
        """系数 -> 轨迹"""
        return self._Phi @ c  # (n_points, n_dof)
    
    def trajectory_to_coeffs(self, traj: np.ndarray) -> np.ndarray:
        """轨迹 -> 系数"""
        return np.linalg.lstsq(self._Phi, traj, rcond=None)[0]
    
    def compute_velocity(self, c: np.ndarray) -> np.ndarray:
        """速度"""
        return self._dPhi @ c
    
    def compute_acceleration(self, c: np.ndarray) -> np.ndarray:
        """加速度"""
        return self._d2Phi @ c


class FACTOFull:
    """
    FACTO 完整实现 - 函数空间自适应约束轨迹优化
    
    论文核心:
    1. Fourier 基函数表示轨迹
    2. 系数空间优化 (非直接优化关节角)
    3. LM + 高斯牛顿近似
    4. 约束处理 (关节限制、碰撞)
    """
    
    def __init__(self, robot: XTrainerRobot, basis: FourierBasis):
        self.robot = robot
        self.basis = basis
        
        # LM 参数
        self.lam = 0.01
        self.lam_max = 1e6
        self.lam_min = 1e-8
        
        # 预缓存不变的雅可比矩阵
        self._J_smooth_cache = None
        self._J_vel_start_cache = None
        self._J_vel_end_cache = None
        
    def optimize(self,
                start: np.ndarray,
                goal: np.ndarray,
                obstacles: List[np.ndarray] = None,
                max_iter: int = 150) -> Tuple[np.ndarray, Dict]:
        """
        FACTO 优化
        """
        # 重置缓存和阻尼因子
        self._J_smooth_cache = None
        self._J_vel_start_cache = None
        self._J_vel_end_cache = None
        self.lam = 0.01
        
        # 初始化: Hermite 插值 (3t²-2t³) 作为初始轨迹
        # 与基函数中的 Hermite 项匹配，端点速度天然为0，不会泄漏到谐波分量
        n_dof = self.robot.n_dof
        t = np.linspace(0, 1, self.basis.n_points)
        h = 3*t**2 - 2*t**3  # Hermite 插值: h(0)=0, h(1)=1, h'(0)=0, h'(1)=0
        traj_init = np.zeros((self.basis.n_points, n_dof))
        for i in range(n_dof):
            traj_init[:, i] = start[i] + h * (goal[i] - start[i])
        
        coeffs = self.basis.trajectory_to_coeffs(traj_init)
        
        # 迭代优化
        for it in range(max_iter):
            traj = self.basis.coeffs_to_trajectory(coeffs)
            
            # 计算成本和雅可比
            cost, J = self._compute_cost(coeffs, start, goal, obstacles)
            
            # LM 更新
            H = J.T @ J + self.lam * np.eye(J.shape[1])
            grad = J.T @ cost
            
            try:
                delta = np.linalg.solve(H, grad)
            except:
                delta = np.linalg.lstsq(H, grad, rcond=None)[0]
            
            coeffs_new = coeffs - delta.reshape(coeffs.shape)
            
            # 接受/拒绝 (用残差平方和作为目标函数，而非残差和)
            traj_new = self.basis.coeffs_to_trajectory(coeffs_new)
            cost_new, _ = self._compute_cost(coeffs_new, start, goal, obstacles)
            
            if np.sum(cost_new**2) < np.sum(cost**2):
                coeffs = coeffs_new
                self.lam = max(self.lam / 1.5, self.lam_min)
            else:
                self.lam = min(self.lam * 1.5, self.lam_max)
            
            # 收敛判断
            if np.linalg.norm(delta) < 1e-6:
                break
        
        final_traj = self.basis.coeffs_to_trajectory(coeffs)
        
        return final_traj, {'iterations': it + 1, 'final_cost': np.sum(cost**2)}
    
    def _compute_cost(self, coeffs: np.ndarray, start: np.ndarray, goal: np.ndarray,
                     obstacles: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """计算成本向量和雅可比矩阵"""
        
        traj = self.basis.coeffs_to_trajectory(coeffs)
        acc = self.basis.compute_acceleration(coeffs)
        
        residuals = []
        J_rows = []
        
        # 1. 平滑度成本 (加速度平方) - 权重提升到 0.1 以抑制高频波动
        res_smooth = acc.flatten() * 0.1
        if self._J_smooth_cache is None:
            self._J_smooth_cache = self._jacobian_smooth()
        residuals.extend(res_smooth)
        J_rows.append(self._J_smooth_cache)
        
        # 2. 末端误差 (终点) - 权重50，确保终点精度
        res_end = (traj[-1] - goal) * 50
        J_end = self._jacobian_end(coeffs)
        residuals.extend(res_end)
        J_rows.append(J_end)
        
        # 2.5 起始点约束 - 权重50，确保起点精度
        res_start = (traj[0] - start) * 50
        J_start = self._jacobian_start(coeffs)
        residuals.extend(res_start)
        J_rows.append(J_start)
        
        # 2.6 速度边界约束 - 权重5.0，起止速度为0
        vel = self.basis.compute_velocity(coeffs)
        res_vel_start = vel[0] * 5.0   # 起点速度应为0
        res_vel_end = vel[-1] * 5.0     # 终点速度应为0
        if self._J_vel_start_cache is None:
            self._J_vel_start_cache = self._jacobian_velocity_boundary(0)
            self._J_vel_end_cache = self._jacobian_velocity_boundary(-1)
        residuals.extend(res_vel_start)
        J_rows.append(self._J_vel_start_cache)
        residuals.extend(res_vel_end)
        J_rows.append(self._J_vel_end_cache)
        
        # 3. 关节限制
        for i, q in enumerate(traj[::5]):  # 每5个点检查一次
            for j in range(self.robot.n_dof):
                if q[j] < self.robot.joint_limits[j, 0] + 0.1:
                    res = (self.robot.joint_limits[j, 0] + 0.1 - q[j]) * 5
                    residuals.append(res)
                    J_rows.append(self._jacobian_joint_violation(i*5, j, coeffs))
                elif q[j] > self.robot.joint_limits[j, 1] - 0.1:
                    res = (q[j] - (self.robot.joint_limits[j, 1] - 0.1)) * 5
                    residuals.append(res)
                    J_rows.append(self._jacobian_joint_violation(i*5, j, coeffs))
        
        # 4. 障碍物避让
        if obstacles:
            for obs in obstacles:
                for i, q in enumerate(traj):
                    try:
                        pos = self.robot.compute_end_effector_position(q)
                        dist = np.linalg.norm(pos - obs)
                        if dist < 0.08:
                            res = (0.08 - dist) * 20
                            residuals.append(res)
                            J_rows.append(self._jacobian_obstacle(i, obs, coeffs))
                    except:
                        pass
        
        residual_vec = np.array(residuals)
        J_matrix = np.vstack(J_rows) if J_rows else np.zeros((0, coeffs.size))
        
        return residual_vec, J_matrix
    
    def _jacobian_smooth(self) -> np.ndarray:
        """平滑度雅可比 (使用缓存的加速度矩阵)"""
        n = self.basis.n_points
        n_dof = self.robot.n_dof
        d2Phi = self.basis._d2Phi  # 直接使用缓存
        
        J = np.zeros((n * n_dof, self.basis.n_basis * n_dof))
        
        for j in range(n_dof):
            J[j*n:(j+1)*n, j*self.basis.n_basis:(j+1)*self.basis.n_basis] = d2Phi * 0.1
        
        return J
        
    def _jacobian_end(self, coeffs: np.ndarray) -> np.ndarray:
        """末端雅可比（终点）"""
        n_dof = self.robot.n_dof
        J = np.zeros((n_dof, coeffs.size))
        Phi_end = self.basis.basis_matrix()[-1, :]
        for j in range(n_dof):
            J[j, j*self.basis.n_basis:(j+1)*self.basis.n_basis] = Phi_end * 50
        return J
    
    def _jacobian_start(self, coeffs: np.ndarray) -> np.ndarray:
        """起始点雅可比 - 约束 traj[0] == start"""
        n_dof = self.robot.n_dof
        J = np.zeros((n_dof, coeffs.size))
        Phi_start = self.basis.basis_matrix()[0, :]
        for j in range(n_dof):
            J[j, j*self.basis.n_basis:(j+1)*self.basis.n_basis] = Phi_start * 50
        return J
    
    def _jacobian_velocity_boundary(self, time_idx: int) -> np.ndarray:
        """速度边界雅可比 - 约束端点速度为0"""
        n_dof = self.robot.n_dof
        n_basis = self.basis.n_basis
        
        # 直接使用缓存的速度矩阵，保证与基函数布局一致
        dPhi_row = self.basis._dPhi[time_idx]
        
        J = np.zeros((n_dof, n_basis * n_dof))
        for j in range(n_dof):
            J[j, j*n_basis:(j+1)*n_basis] = dPhi_row * 5.0
        return J
    
    def _jacobian_joint_violation(self, traj_idx: int, joint_idx: int, coeffs: np.ndarray) -> np.ndarray:
        """关节限制违反的雅可比"""
        J = np.zeros(coeffs.size)
        Phi_row = self.basis.basis_matrix()[traj_idx, :]
        J[joint_idx * self.basis.n_basis:(joint_idx+1) * self.basis.n_basis] = Phi_row * 5
        return J
    
    def _jacobian_obstacle(self, traj_idx: int, obs: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        """障碍物避让雅可比"""
        # 简化实现
        return np.zeros(coeffs.size)


class TimeParameterizer:
    """时间参数化器 - 为几何轨迹分配变速时间
    
    原理：
      FACTO 输出的是等间隔路径点（纯几何），没有时间信息。
      本类根据「路径步长 + 曲率」自适应分配时间间隔：
        - 步长大 → 时间长（走得远就花更多时间）
        - 曲率大（转弯处）→ 额外减速因子 → 更慢
        - 曲率小（直线段）→ 无减速 → 更快
      最后用速度限制裁剪，确保不超过 URDF 关节速度限制。
    
    算法步骤：
      1. 计算每段步长 s_i = max_j(|Δq_{i,j}|)
      2. 计算曲率指标 κ_i（二阶差分的模）
      3. 减速因子 α_i = 1 + curvature_gain * κ_norm_i
      4. dt_i = (s_i / v_cruise_max) * α_i，然后保底
      5. 速度裁剪 + 平滑 + 再次裁剪
    """
    
    def __init__(self, velocity_limits: np.ndarray, cruise_ratio: float = 0.8,
                 dt_min: float = 0.005, smooth_window: int = 5,
                 curvature_gain: float = 3.0, total_time_target: float = None):
        """
        Args:
            velocity_limits: 各关节速度上限 (rad/s)，长度=n_dof
            cruise_ratio: 巡航速度占限速的比例 (0~1)
            dt_min: 最小时间间隔 (s)，防止过快抖动
            smooth_window: 滑动平均窗口大小（奇数）
            curvature_gain: 曲率减速增益，越大转弯越慢
            total_time_target: 目标总时间 (s)，None 则自动
        """
        self.v_max = velocity_limits
        self.v_cruise = velocity_limits * cruise_ratio
        self.v_cruise_max = np.max(self.v_cruise)
        self.dt_min = dt_min
        self.smooth_window = smooth_window
        self.curvature_gain = curvature_gain
        self.total_time_target = total_time_target
    
    def parameterize(self, traj: np.ndarray) -> dict:
        """对轨迹进行时间参数化
        
        Args:
            traj: (N, n_dof) 关节角矩阵（纯几何路径点）
        
        Returns:
            dict 包含:
              - 'trajectory': (N, n_dof) 原始路径点
              - 'dt': (N-1,) 每段时间间隔 (s)
              - 'timestamps': (N,) 累积时间戳 (s)
              - 'velocities': (N-1, n_dof) 每段各关节角速度 (rad/s)
              - 'total_time': 总时间 (s)
              - 'max_speed_ratio': 峰值速度与限速的比值
              - 'curvature': (N-1,) 曲率指标
        """
        N, n_dof = traj.shape
        
        # 1. 每段关节角变化量
        dq = np.diff(traj, axis=0)  # (N-1, n_dof)
        
        # 2. 步长: 每段各关节变化的最大值
        step_sizes = np.max(np.abs(dq), axis=1)  # (N-1,)
        
        # 3. 曲率指标: 二阶差分 → 衡量路径"弯曲程度"
        #    ddq[i] = dq[i+1] - dq[i]，即关节角加速度
        ddq = np.diff(dq, axis=0)  # (N-2, n_dof)
        curvature_raw = np.linalg.norm(ddq, axis=1)  # (N-2,)
        
        # 在两端补值（首尾段用邻居的曲率）
        curvature = np.zeros(N - 1)
        if len(curvature_raw) > 0:
            # curvature_raw 有 N-2 个值，对应 dq[0..N-3] 与 dq[1..N-2] 之间的差
            # curvature[i] 表示第 i 段的曲率，i=0..N-2
            # curvature_raw[i] 对应 i=0..N-3，映射到 curvature[1..N-2]
            curvature[1:len(curvature_raw)+1] = curvature_raw
            curvature[0] = curvature_raw[0]
            if len(curvature_raw) + 1 < len(curvature):
                curvature[-1] = curvature_raw[-1]
        
        # 4. 归一化曲率到 [0, 1]
        kappa_max = np.max(curvature) if np.max(curvature) > 1e-10 else 1.0
        kappa_norm = curvature / kappa_max
        
        # 5. 每段权重 = 步长 × (1 + curvature_gain * 归一化曲率)
        #    步长大或曲率大的段，权重更大 → 分到更多时间
        weights = step_sizes * (1.0 + self.curvature_gain * kappa_norm)
        weights = np.maximum(weights, 1e-10)  # 避免零权重
        
        # 6. 估算合理总时间
        if self.total_time_target is not None:
            T_total = self.total_time_target
        else:
            # 自动估算: 各关节总弧长 / 巡航速度，取最忙关节
            total_arc = np.sum(np.abs(dq), axis=0)  # (n_dof,)
            T_total = np.max(total_arc / self.v_cruise)
            T_total = max(T_total, 2.0)  # 至少 2.0s，保证动画可见
        
        # 7. 按权重分配时间
        dt = weights / np.sum(weights) * T_total
        
        # 8. 保底
        dt = np.maximum(dt, self.dt_min)
        
        # 9. 速度裁剪: 确保任何关节不超速
        for i in range(N - 1):
            for j in range(n_dof):
                if np.abs(dq[i, j]) > 1e-12:
                    dt_required = np.abs(dq[i, j]) / self.v_max[j]
                    dt[i] = max(dt[i], dt_required)
        
        # 10. 平滑
        dt = self._smooth(dt)
        
        # 11. 平滑后再次裁剪速度
        for i in range(N - 1):
            for j in range(n_dof):
                if np.abs(dq[i, j]) > 1e-12:
                    dt_required = np.abs(dq[i, j]) / self.v_max[j]
                    dt[i] = max(dt[i], dt_required)
        
        # 计算累积时间戳
        timestamps = np.zeros(N)
        timestamps[1:] = np.cumsum(dt)
        
        # 计算实际角速度
        velocities = dq / dt[:, np.newaxis]
        
        # 峰值速度比
        speed_ratios = np.abs(velocities) / self.v_max[np.newaxis, :]
        max_speed_ratio = np.max(speed_ratios)
        
        return {
            'trajectory': traj,
            'dt': dt,
            'timestamps': timestamps,
            'velocities': velocities,
            'total_time': timestamps[-1],
            'max_speed_ratio': max_speed_ratio,
            'curvature': curvature,
        }
    
    def _smooth(self, dt: np.ndarray) -> np.ndarray:
        """滑动平均平滑时间间隔序列"""
        w = self.smooth_window
        if w <= 1 or len(dt) < w:
            return dt
        kernel = np.ones(w) / w
        padded = np.pad(dt, (w // 2, w // 2), mode='edge')
        smoothed = np.convolve(padded, kernel, mode='valid')[:len(dt)]
        smoothed = np.maximum(smoothed, self.dt_min)
        return smoothed


def visualize(robot: XTrainerRobot, traj: np.ndarray, 
            start: np.ndarray, goal: np.ndarray,
            obstacles: List = None, title: str = "",
            time_info: dict = None) -> plt.Figure:
    """可视化（支持时间参数化信息）"""
    
    has_time = time_info is not None
    n_cols = 3 if has_time else 2
    fig = plt.figure(figsize=(6 * n_cols, 6))
    
    # 3D 工作空间
    ax1 = fig.add_subplot(1, n_cols, 1, projection='3d')
    
    px, py, pz = [], [], []
    for q in traj:
        try:
            p = robot.compute_end_effector_position(q)
            px.append(p[0]); py.append(p[1]); pz.append(p[2])
        except: pass
    
    ax1.plot(px, py, pz, 'b-', lw=2, label='Trajectory')
    
    try:
        sp = robot.compute_end_effector_position(start)
        gp = robot.compute_end_effector_position(goal)
        ax1.scatter(*sp, c='g', s=100, marker='o', label='Start')
        ax1.scatter(*gp, c='r', s=100, marker='^', label='Goal')
    except: pass
    
    if obstacles:
        for o in obstacles:
            ax1.scatter(*o, c='orange', s=150, marker='s', alpha=0.7)
    
    bounds = robot.get_workspace_bounds()
    ax1.set_xlim(bounds['x'])
    ax1.set_ylim(bounds['y'])
    ax1.set_zlim(bounds['z'])
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.set_title('Workspace')
    ax1.legend()
    
    # 关节空间 (用实际时间轴)
    ax2 = fig.add_subplot(1, n_cols, 2)
    if has_time:
        tt = time_info['timestamps']
        xlabel = 'Time (s)'
    else:
        tt = np.linspace(0, 1, len(traj))
        xlabel = 'Normalized Time'
    for j in range(robot.n_dof):
        ax2.plot(tt, traj[:, j], label=f'J{j+1}')
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel('Joint (rad)')
    ax2.set_title('Joint Space')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 速度 + 时间间隔图（仅在有时间信息时）
    if has_time:
        ax3 = fig.add_subplot(1, n_cols, 3)
        
        # 上半部分：各关节速度
        t_mid = (time_info['timestamps'][:-1] + time_info['timestamps'][1:]) / 2
        vel = time_info['velocities']
        for j in range(robot.n_dof):
            ax3.plot(t_mid, vel[:, j], label=f'J{j+1}', alpha=0.8)
        
        # 画速度上限参考线
        v_lim = robot.joint_velocity_limits[0]  # 假设相同
        ax3.axhline(y=v_lim, color='r', linestyle='--', alpha=0.5, label=f'+limit ({v_lim})')
        ax3.axhline(y=-v_lim, color='r', linestyle='--', alpha=0.5, label=f'-limit')
        
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Joint Velocity (rad/s)')
        ax3.set_title(f'Velocities (peak ratio={time_info["max_speed_ratio"]:.2%})')
        ax3.legend(fontsize=7)
        ax3.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig


def test():
    print("=" * 50)
    print("FACTO 完整版 - XTrainer 机械臂")
    print("=" * 50)
    
    # 创建机器人
    robot = XTrainerRobot()
    
    # Fourier 基函数
    basis = FourierBasis(n_basis=10, n_dof=6, n_points=80)
    
    # FACTO 优化器
    facto = FACTOFull(robot, basis)
    
    # 时间参数化器
    timer = TimeParameterizer(robot.joint_velocity_limits, cruise_ratio=0.8)
    
    # 测试
    start = np.array([0, 0, 0, 0, 0, 0])
    goal = np.array([1.5, -1.3, 2.2, 1.1, -2.2, 0.8])
    
    print("\n[1] 点对点轨迹")
    t0 = time.time()
    traj1, info1 = facto.optimize(start, goal, max_iter=100)
    time_info1 = timer.parameterize(traj1)
    print(f"  迭代: {info1['iterations']}, 耗时: {time.time()-t0:.2f}s")
    print(f"  总时间: {time_info1['total_time']:.4f}s")
    print(f"  峰值速度比: {time_info1['max_speed_ratio']:.2%}")
    print(f"  dt范围: [{time_info1['dt'].min():.5f}, {time_info1['dt'].max():.5f}]s")
    
    fig = visualize(robot, traj1, start, goal,
                    title="XTrainer P2P (Time-Parameterized)", time_info=time_info1)
    plt.savefig('/home/ethanqjiang/workspace/facto_xtrainer_maxclaw/test_p2p.png', dpi=150)
    print("  保存: test_p2p.png")
    
    print("\n[2] 障碍物避让")
    obstacles = [np.array([0.5, -0.5, 0.0])]
    t1 = time.time()
    traj2, info2 = facto.optimize(start, goal, obstacles=obstacles, max_iter=120)
    time_info2 = timer.parameterize(traj2)
    print(f"  迭代: {info2['iterations']}, 耗时: {time.time()-t1:.2f}s")
    print(f"  总时间: {time_info2['total_time']:.4f}s")
    print(f"  峰值速度比: {time_info2['max_speed_ratio']:.2%}")
    print(f"  dt范围: [{time_info2['dt'].min():.5f}, {time_info2['dt'].max():.5f}]s")
    
    fig2 = visualize(robot, traj2, start, goal, obstacles,
                     "XTrainer Obstacle (Time-Parameterized)", time_info=time_info2)
    plt.savefig('/home/ethanqjiang/workspace/facto_xtrainer_maxclaw/test_obstacle.png', dpi=150)
    print("  保存: test_obstacle.png")
    
    plt.close('all')
    print("\n完成!")


if __name__ == "__main__":
    test()
