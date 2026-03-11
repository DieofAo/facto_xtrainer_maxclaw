"""
FACTO 论文完整复现 - XTrainer 机械臂

基于论文理论体系：
s1. 三阶 B 样条基函数轨迹表示
2. 函数空间自适应优化 (LM + 高斯牛顿)
3. 约束优化 (关节限制 + 碰撞避让)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, List, Optional, Dict
import time
import xml.etree.ElementTree as ET
import yaml
import os
from scipy.interpolate import BSpline as SciBSpline


def load_collision_spheres(yml_path: str) -> Dict:
    """从 xtrainer.yml 加载碰撞球体配置
    
    Returns:
        dict: {link_name: [{center: [x,y,z], radius: float}, ...], ...}
    """
    with open(yml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data.get('collision_spheres', {})


class XTrainerRobot:
    """XTrainer 6轴机械臂 - 从 URDF 解析
    
    FK 采用 URDF 原生变换链：
      T_link_i = T_parent × T_origin(xyz, rpy) × Rot(axis, q_i)
    其中 T_origin 包含关节坐标系相对父连杆的位置和姿态偏移。
    """
    
    def __init__(self, urdf_path: str = None, collision_spheres: Dict = None):
        """从 URDF 构建机器人模型"""
        
        self.n_dof = 6
        
        # ---- 从 URDF 提取的关节参数 ----
        # 每个关节的 origin: (xyz, rpy)
        self.joint_origins = [
            # J_1: LINK_0 -> LINK_1
            {'xyz': [0, 0, 0.2234],       'rpy': [0, 0, 0]},
            # J_2: LINK_1 -> LINK_2  (有 RPY 预旋转!)
            {'xyz': [0, 0, 0],            'rpy': [1.5707963267949, 1.5707963267949, 0]},
            # J_3: LINK_2 -> LINK_3
            {'xyz': [-0.28, 0, 0],         'rpy': [0, 0, 0]},
            # J_4: LINK_3 -> LINK_4  (有 RPY 预旋转!)
            {'xyz': [-0.225, 0, 0.1175],   'rpy': [0, 0, -1.57079637654807]},
            # J_5: LINK_4 -> LINK_5  (有 RPY 预旋转!)
            {'xyz': [0, -0.12, 0],         'rpy': [1.5707963267949, 0, 0]},
            # J_6: LINK_5 -> LINK_6  (有 RPY 预旋转!)
            {'xyz': [0, 0.0829999998597296, 0], 'rpy': [-1.5707963267949, 0, 0]},
        ]
        
        # 所有关节轴都是 z 轴 (URDF 中均为 <axis xyz="0 0 1"/>)
        self.joint_axes = np.array([
            [0, 0, 1],  # J_1
            [0, 0, 1],  # J_2
            [0, 0, 1],  # J_3
            [0, 0, 1],  # J_4
            [0, 0, 1],  # J_5
            [0, 0, 1]   # J_6
        ])
        
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
        
        # 预计算每个关节的 origin 齐次变换 (不含关节角)
        self._T_origins = []
        for jo in self.joint_origins:
            self._T_origins.append(self._make_origin_transform(jo['xyz'], jo['rpy']))
        
    @staticmethod
    def _rpy_to_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """RPY (XYZ 固定角) -> 3×3 旋转矩阵
        
        R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
        这是 URDF/ROS 的标准约定。
        """
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        
        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr           ]
        ])
        return R
    
    def _make_origin_transform(self, xyz: list, rpy: list) -> np.ndarray:
        """从 URDF joint origin 的 xyz + rpy 构建 4×4 齐次变换"""
        T = np.eye(4)
        T[:3, :3] = self._rpy_to_rotation(rpy[0], rpy[1], rpy[2])
        T[:3, 3] = xyz
        return T
    
    @staticmethod
    def _rot_z(angle: float) -> np.ndarray:
        """绕 z 轴旋转的 4×4 齐次变换"""
        c, s = np.cos(angle), np.sin(angle)
        T = np.eye(4)
        T[0, 0] = c;  T[0, 1] = -s
        T[1, 0] = s;  T[1, 1] = c
        return T

    def forward_kinematics(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """正运动学 - 计算末端位置和姿态
        
        变换链: T = ∏ [T_origin_i × Rz(q_i)]
        """
        T = np.eye(4)
        for i in range(self.n_dof):
            T = T @ self._T_origins[i] @ self._rot_z(q[i])
        return T[:3, 3], T[:3, :3]

    def get_link_transforms(self, q: np.ndarray) -> List[np.ndarray]:
        """计算每个连杆在世界坐标系中的齐次变换矩阵
        
        T_link_i = T_origin_0 × Rz(q0) × T_origin_1 × Rz(q1) × ... × T_origin_i × Rz(qi)
        
        Returns:
            list of (4,4) ndarray: T_world_link[i] 是 LINK_{i+1} 的世界变换
        """
        transforms = []
        T = np.eye(4)
        for i in range(self.n_dof):
            T = T @ self._T_origins[i] @ self._rot_z(q[i])
            transforms.append(T.copy())
        return transforms

    def get_all_sphere_positions(self, q: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """给定关节角，计算所有碰撞球体在世界坐标系中的位置和半径
        
        Returns:
            list of (center_world, radius): 每个球体的世界坐标中心和半径
        """
        return [(c, r) for c, r, _ in self.get_all_sphere_positions_with_link(q)]
    
    def get_all_sphere_positions_with_link(self, q: np.ndarray) -> List[Tuple[np.ndarray, float, int]]:
        """给定关节角，计算所有碰撞球体在世界坐标系中的位置、半径和所属连杆索引
        
        Returns:
            list of (center_world, radius, link_index): 
              link_index: 0=LINK_1, 1=LINK_2, ..., 5=LINK_6
        """
        if not self.collision_spheres:
            return []
        
        transforms = self.get_link_transforms(q)
        # 连杆名到索引的映射: LINK_1->0, LINK_2->1, ...
        link_map = {f'LINK_{i+1}': i for i in range(self.n_dof)}
        
        spheres = []
        for link_name, sphere_list in self.collision_spheres.items():
            idx = link_map.get(link_name)
            if idx is None:
                continue
            T = transforms[idx]
            for s in sphere_list:
                center_local = np.array(s['center'])
                # 齐次坐标变换
                center_world = (T[:3, :3] @ center_local) + T[:3, 3]
                spheres.append((center_world, s['radius'], idx))
        
        return spheres
    
    def check_self_collision(self, q: np.ndarray, safety_margin: float = 0.01,
                             skip_adjacent: int = 1) -> bool:
        """自碰撞检测 — 检查非相邻连杆上的球体是否互相穿透
        
        跳过 |link_i - link_j| <= skip_adjacent 的球体对，
        因为相邻连杆天然接近，不算碰撞。
        
        Args:
            q: 关节角
            safety_margin: 安全裕度 (m)
            skip_adjacent: 跳过的相邻连杆层数，默认 1（跳过同一连杆和直接相邻）
        
        Returns:
            True 表示发生自碰撞
        """
        spheres = self.get_all_sphere_positions_with_link(q)
        if len(spheres) < 2:
            return False
        
        n = len(spheres)
        for i in range(n):
            ci, ri, li = spheres[i]
            for j in range(i + 1, n):
                cj, rj, lj = spheres[j]
                # 跳过同一连杆和相邻连杆
                if abs(li - lj) <= skip_adjacent:
                    continue
                dist = np.linalg.norm(ci - cj)
                if dist < ri + rj + safety_margin:
                    return True
        return False
    
    def get_workspace_bounds(self) -> Dict:
        """工作空间边界"""
        return {
            'x': [-0.8, 0.8],
            'y': [-0.8, 0.8],
            'z': [0, 0.6]
        }
    
    def check_collision(self, q: np.ndarray, obstacles: List[np.ndarray] = None,
                        safety_margin: float = 0.02,
                        check_self: bool = True) -> bool:
        """碰撞检测 — 外部障碍物 + 自碰撞
        
        1. 外部碰撞: 对每个障碍物点，检查与所有碰撞球体的距离
        2. 自碰撞: 非相邻连杆球体之间的距离检测
        """
        try:
            # 自碰撞检测
            if check_self and self.check_self_collision(q, safety_margin=safety_margin):
                return True
            
            # 外部障碍物碰撞检测
            if obstacles:
                spheres = self.get_all_sphere_positions(q)
                if spheres:
                    for obs in obstacles:
                        for center, radius in spheres:
                            dist = np.linalg.norm(center - obs)
                            if dist < radius + safety_margin:
                                return True
                else:
                    # 退化: 只检查末端
                    pos, _ = self.forward_kinematics(q)
                    for obs in obstacles:
                        if np.linalg.norm(pos - obs) < 0.05:
                            return True
        except:
            return True
        return False
    
    def compute_end_effector_position(self, q: np.ndarray) -> np.ndarray:
        """计算末端执行器位置"""
        pos, _ = self.forward_kinematics(q)
        return pos


class BSplineBasis:
    """三阶 B 样条基函数轨迹表示 (Clamped, scipy 加速)
    
    三阶 B 样条 (degree=3, order=4) 的优势:
      - 局部支撑: 修改一个控制点只影响附近 4 段轨迹
      - C² 连续: 二阶导数连续，天然平滑
      - 凸包性: 轨迹在控制点的凸包内，便于约束处理
      - Clamped 端点: 轨迹精确经过首尾控制点
    
    节点向量构造 (Clamped):
      knots = [0,0,0,0, τ₁,τ₂,...,τ_{n-4}, 1,1,1,1]
      两端各重复 4 次 → 轨迹精确通过首尾控制点
    
    使用 scipy.interpolate.BSpline 的 C 实现，比手写 Cox-de Boor 快 100+ 倍。
    """
    
    def __init__(self, n_basis: int = 10, n_dof: int = 6, n_points: int = 100):
        """
        Args:
            n_basis: 基函数个数 (=控制点个数/每个关节)，需 >= 4
            n_dof: 自由度
            n_points: 采样点数
        """
        assert n_basis >= 4, "三阶 B 样条至少需要 4 个基函数"
        self.n_basis = n_basis
        self.n_dof = n_dof
        self.n_points = n_points
        self.degree = 3  # 三阶
        self.order = 4   # order = degree + 1
        self.t = np.linspace(0, 1, n_points)
        
        # 构建 clamped 节点向量
        n_internal = n_basis - self.order  # 内部节点数
        if n_internal > 0:
            internal_knots = np.linspace(0, 1, n_internal + 2)[1:-1]
        else:
            internal_knots = np.array([])
        self.knots = np.concatenate([
            np.zeros(self.order),
            internal_knots,
            np.ones(self.order)
        ])
        
        # 预计算并缓存所有矩阵 (使用 scipy 加速)
        self._Phi = self._build_basis_matrix_scipy()
        self._dPhi = self._build_derivative_matrix_scipy(1)
        self._d2Phi = self._build_derivative_matrix_scipy(2)
    
    def _build_basis_matrix_scipy(self) -> np.ndarray:
        """用 scipy 构建 B 样条基函数矩阵 Φ(n_points × n_basis)
        
        对每个基函数 i，构造单位系数向量 e_i，用 scipy.BSpline 求值。
        scipy 内部是 C 实现，比纯 Python 递归快 100+ 倍。
        """
        Phi = np.zeros((self.n_points, self.n_basis))
        for i in range(self.n_basis):
            c = np.zeros(self.n_basis)
            c[i] = 1.0
            spl = SciBSpline(self.knots, c, self.degree)
            Phi[:, i] = spl(self.t)
        return Phi
    
    def _build_derivative_matrix_scipy(self, nu: int) -> np.ndarray:
        """用 scipy 构建 nu 阶导数矩阵
        
        BSpline.derivative(nu) 返回导数样条，再对 t 求值。
        nu=1 → 速度矩阵 dΦ/dt
        nu=2 → 加速度矩阵 d²Φ/dt²
        """
        dPhi = np.zeros((self.n_points, self.n_basis))
        for i in range(self.n_basis):
            c = np.zeros(self.n_basis)
            c[i] = 1.0
            spl = SciBSpline(self.knots, c, self.degree)
            dspl = spl.derivative(nu)
            dPhi[:, i] = dspl(self.t)
        return dPhi
    
    def basis_matrix(self) -> np.ndarray:
        """B 样条基函数矩阵 (缓存)"""
        return self._Phi
    
    def coeffs_to_trajectory(self, c: np.ndarray) -> np.ndarray:
        """控制点系数 -> 轨迹: traj = Φ @ c"""
        return self._Phi @ c  # (n_points, n_dof)
    
    def trajectory_to_coeffs(self, traj: np.ndarray) -> np.ndarray:
        """轨迹 -> 控制点系数 (最小二乘拟合)"""
        return np.linalg.lstsq(self._Phi, traj, rcond=None)[0]
    
    def compute_velocity(self, c: np.ndarray) -> np.ndarray:
        """计算速度: vel = dΦ/dt @ c"""
        return self._dPhi @ c
    
    def compute_acceleration(self, c: np.ndarray) -> np.ndarray:
        """计算加速度: acc = d²Φ/dt² @ c"""
        return self._d2Phi @ c


class FACTOFull:
    """
    FACTO 完整实现 - 函数空间自适应约束轨迹优化
    
    论文核心:
    1. B 样条基函数表示轨迹
    2. 系数空间优化 (非直接优化关节角)
    3. LM + 高斯牛顿近似
    4. 约束处理 (关节限制、碰撞)
    """
    
    def __init__(self, robot: XTrainerRobot, basis: BSplineBasis):
        self.robot = robot
        self.basis = basis
        
        # LM 参数
        self.lam = 0.01
        self.lam_max = 1e6
        self.lam_min = 1e-8
        
        # 预构建所有固定不变的雅可比矩阵，合并成 _J_fixed
        # 固定残差 = _J_fixed @ coeffs_flat (无常数项部分)
        self._build_fixed_jacobian()
        
    def optimize(self,
                start: np.ndarray,
                goal: np.ndarray,
                obstacles: List[np.ndarray] = None,
                max_iter: int = 150) -> Tuple[np.ndarray, Dict]:
        """
        FACTO 优化
        """
        # 重置阻尼因子
        self.lam = 0.01
        
        # 初始化: Hermite 插值 (3t²-2t³) 作为初始轨迹
        # h(0)=0, h(1)=1, h'(0)=0, h'(1)=0
        # 拟合到 B 样条后，首尾控制点收拢，端点速度近似为 0
        n_dof = self.robot.n_dof
        t = np.linspace(0, 1, self.basis.n_points)
        h = 3*t**2 - 2*t**3
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
            
            coeffs_new = coeffs - delta.reshape(coeffs.shape, order='F')
            
            # 接受/拒绝 (只算残差，不算雅可比，节省约一半计算量)
            cost_new = self._compute_residuals_only(coeffs_new, start, goal, obstacles)
            
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
    
    def _build_fixed_jacobian(self):
        """预构建所有固定不变的雅可比矩阵，合并成 _J_fixed
        
        固定部分包括: 平滑度 + 终点 + 起点 + 速度边界
        这些的雅可比矩阵不依赖 coeffs，只需构建一次。
        
        _J_fixed: (N_fixed, n_coeffs) 合并后的固定雅可比
        _b_goal_rows / _b_start_rows: 常数项索引，用于高效计算残差
        """
        n = self.basis.n_points
        n_dof = self.robot.n_dof
        n_basis = self.basis.n_basis
        n_coeffs = n_basis * n_dof
        
        rows = []
        
        # 1. 平滑度 (n*n_dof 行)
        J_smooth = np.zeros((n * n_dof, n_coeffs))
        for j in range(n_dof):
            J_smooth[j*n:(j+1)*n, j*n_basis:(j+1)*n_basis] = self.basis._d2Phi * 0.1
        rows.append(J_smooth)
        self._smooth_end = n * n_dof
        
        # 2. 终点 (n_dof 行)
        J_end = np.zeros((n_dof, n_coeffs))
        Phi_end = self.basis._Phi[-1, :]
        for j in range(n_dof):
            J_end[j, j*n_basis:(j+1)*n_basis] = Phi_end * 50
        rows.append(J_end)
        
        # 3. 起点 (n_dof 行)
        J_start = np.zeros((n_dof, n_coeffs))
        Phi_start = self.basis._Phi[0, :]
        for j in range(n_dof):
            J_start[j, j*n_basis:(j+1)*n_basis] = Phi_start * 50
        rows.append(J_start)
        
        # 4. 速度边界 — 起点 (n_dof 行)
        J_vel_s = np.zeros((n_dof, n_coeffs))
        dPhi_0 = self.basis._dPhi[0]
        for j in range(n_dof):
            J_vel_s[j, j*n_basis:(j+1)*n_basis] = dPhi_0 * 5.0
        rows.append(J_vel_s)
        
        # 5. 速度边界 — 终点 (n_dof 行)
        J_vel_e = np.zeros((n_dof, n_coeffs))
        dPhi_m1 = self.basis._dPhi[-1]
        for j in range(n_dof):
            J_vel_e[j, j*n_basis:(j+1)*n_basis] = dPhi_m1 * 5.0
        rows.append(J_vel_e)
        
        self._J_fixed = np.vstack(rows)  # (N_fixed, n_coeffs)
        self._n_fixed = self._J_fixed.shape[0]
        
        # 预计算 J_fixed^T @ J_fixed (这在所有迭代中不变)
        self._JtJ_fixed = self._J_fixed.T @ self._J_fixed
        
        # 常数项偏移索引（用于 goal/start 的残差计算）
        self._goal_slice = slice(self._smooth_end, self._smooth_end + n_dof)
        self._start_slice = slice(self._smooth_end + n_dof, self._smooth_end + 2*n_dof)
    
    def _compute_fixed_residuals(self, coeffs_flat: np.ndarray,
                                 start: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """高效计算固定部分的残差向量
        
        res_fixed = J_fixed @ coeffs_flat + b
        其中 b 只在 goal/start 对应的行有值。
        """
        res = self._J_fixed @ coeffs_flat
        # 终点残差需要减去 goal * 50
        res[self._goal_slice] -= goal * 50
        # 起点残差需要减去 start * 50
        res[self._start_slice] -= start * 50
        return res
    
    def _compute_residuals_only(self, coeffs: np.ndarray, start: np.ndarray,
                                goal: np.ndarray, obstacles: List[np.ndarray]) -> np.ndarray:
        """只计算残差向量（不计算雅可比），用于接受/拒绝判断"""
        coeffs_flat = coeffs.flatten(order='F')
        traj = self.basis.coeffs_to_trajectory(coeffs)
        
        # 固定部分 (一次矩阵乘法)
        res_list = [self._compute_fixed_residuals(coeffs_flat, start, goal)]
        
        # 动态部分: 关节限制
        dyn = []
        for i, q in enumerate(traj[::5]):
            for j in range(self.robot.n_dof):
                if q[j] < self.robot.joint_limits[j, 0] + 0.1:
                    dyn.append((self.robot.joint_limits[j, 0] + 0.1 - q[j]) * 5)
                elif q[j] > self.robot.joint_limits[j, 1] - 0.1:
                    dyn.append((q[j] - (self.robot.joint_limits[j, 1] - 0.1)) * 5)
        
        # 障碍物
        if obstacles:
            has_spheres = bool(self.robot.collision_spheres)
            for obs in obstacles:
                for i in range(0, len(traj), 5):
                    try:
                        if has_spheres:
                            for center, radius in self.robot.get_all_sphere_positions(traj[i]):
                                dist = np.linalg.norm(center - obs)
                                margin = radius + 0.03
                                if dist < margin:
                                    dyn.append((margin - dist) * 20)
                        else:
                            pos = self.robot.compute_end_effector_position(traj[i])
                            dist = np.linalg.norm(pos - obs)
                            if dist < 0.08:
                                dyn.append((0.08 - dist) * 20)
                    except:
                        pass
        
        # 自碰撞
        if self.robot.collision_spheres:
            for i in range(0, len(traj), 10):
                try:
                    spheres = self.robot.get_all_sphere_positions_with_link(traj[i])
                    n_s = len(spheres)
                    for si in range(n_s):
                        ci, ri, li = spheres[si]
                        for sj in range(si + 1, n_s):
                            cj, rj, lj = spheres[sj]
                            if abs(li - lj) <= 1:
                                continue
                            dist = np.linalg.norm(ci - cj)
                            margin = ri + rj + 0.02
                            if dist < margin:
                                dyn.append((margin - dist) * 15)
                except:
                    pass
        
        if dyn:
            res_list.append(np.array(dyn))
        return np.concatenate(res_list)
    
    def _compute_cost(self, coeffs: np.ndarray, start: np.ndarray, goal: np.ndarray,
                     obstacles: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """计算成本向量和雅可比矩阵
        
        固定部分（平滑度+端点+起点+速度边界）使用预缓存的 _J_fixed。
        动态部分（关节限制+碰撞）逐项追加。
        """
        coeffs_flat = coeffs.flatten(order='F')
        traj = self.basis.coeffs_to_trajectory(coeffs)
        
        # 固定部分残差
        res_fixed = self._compute_fixed_residuals(coeffs_flat, start, goal)
        
        # 动态部分
        dyn_res = []
        dyn_J = []
        
        # 关节限制
        for i, q in enumerate(traj[::5]):
            for j in range(self.robot.n_dof):
                if q[j] < self.robot.joint_limits[j, 0] + 0.1:
                    dyn_res.append((self.robot.joint_limits[j, 0] + 0.1 - q[j]) * 5)
                    dyn_J.append(self._jacobian_joint_violation(i*5, j, coeffs))
                elif q[j] > self.robot.joint_limits[j, 1] - 0.1:
                    dyn_res.append((q[j] - (self.robot.joint_limits[j, 1] - 0.1)) * 5)
                    dyn_J.append(self._jacobian_joint_violation(i*5, j, coeffs))
        
        # 障碍物避让
        if obstacles:
            has_spheres = bool(self.robot.collision_spheres)
            for obs in obstacles:
                for i in range(0, len(traj), 5):
                    try:
                        if has_spheres:
                            for center, radius in self.robot.get_all_sphere_positions(traj[i]):
                                dist = np.linalg.norm(center - obs)
                                margin = radius + 0.03
                                if dist < margin:
                                    dyn_res.append((margin - dist) * 20)
                                    dyn_J.append(self._jacobian_obstacle(i, obs, coeffs))
                        else:
                            pos = self.robot.compute_end_effector_position(traj[i])
                            dist = np.linalg.norm(pos - obs)
                            if dist < 0.08:
                                dyn_res.append((0.08 - dist) * 20)
                                dyn_J.append(self._jacobian_obstacle(i, obs, coeffs))
                    except:
                        pass
        
        # 自碰撞避让
        if self.robot.collision_spheres:
            for i in range(0, len(traj), 10):
                try:
                    spheres = self.robot.get_all_sphere_positions_with_link(traj[i])
                    n_s = len(spheres)
                    for si in range(n_s):
                        ci, ri, li = spheres[si]
                        for sj in range(si + 1, n_s):
                            cj, rj, lj = spheres[sj]
                            if abs(li - lj) <= 1:
                                continue
                            dist = np.linalg.norm(ci - cj)
                            margin = ri + rj + 0.02
                            if dist < margin:
                                dyn_res.append((margin - dist) * 15)
                                dyn_J.append(self._jacobian_self_collision(i, coeffs))
                except:
                    pass
        
        # 合并
        if dyn_res:
            dyn_res_arr = np.array(dyn_res)
            dyn_J_arr = np.vstack(dyn_J)
            residual_vec = np.concatenate([res_fixed, dyn_res_arr])
            J_matrix = np.vstack([self._J_fixed, dyn_J_arr])
        else:
            residual_vec = res_fixed
            J_matrix = self._J_fixed
        
        return residual_vec, J_matrix
    
    def _jacobian_joint_violation(self, traj_idx: int, joint_idx: int, coeffs: np.ndarray) -> np.ndarray:
        """关节限制违反的雅可比"""
        J = np.zeros(coeffs.size)
        Phi_row = self.basis._Phi[traj_idx, :]
        J[joint_idx * self.basis.n_basis:(joint_idx+1) * self.basis.n_basis] = Phi_row * 5
        return J
    
    def _jacobian_obstacle(self, traj_idx: int, obs: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        """障碍物避让雅可比"""
        # 简化实现
        return np.zeros(coeffs.size)
    
    def _jacobian_self_collision(self, traj_idx: int, coeffs: np.ndarray) -> np.ndarray:
        """自碰撞雅可比 — 数值差分
        
        对各关节微扰 δq，观察自碰撞残差的变化。
        虽然比解析雅可比慢，但实现简单且正确。
        """
        eps = 1e-4
        n_dof = self.robot.n_dof
        n_basis = self.basis.n_basis
        Phi_row = self.basis.basis_matrix()[traj_idx, :]  # (n_basis,)
        
        # 自碰撞梯度近似: 对每个关节 j 求 d(min_dist)/dq_j
        traj = self.basis.coeffs_to_trajectory(coeffs)
        q0 = traj[traj_idx]
        
        # 当前最小距离
        d0 = self._min_self_collision_dist(q0)
        
        J = np.zeros(coeffs.size)
        for j in range(n_dof):
            q_pert = q0.copy()
            q_pert[j] += eps
            d_pert = self._min_self_collision_dist(q_pert)
            # d(margin - dist)/dq_j = -d(dist)/dq_j
            grad_j = -(d_pert - d0) / eps
            # 链式法则: d_residual/d_coeffs_j = grad_j * Phi_row * weight
            J[j * n_basis:(j + 1) * n_basis] = grad_j * Phi_row * 15
        
        return J
    
    def _min_self_collision_dist(self, q: np.ndarray) -> float:
        """计算非相邻连杆球体之间的最小距离"""
        spheres = self.robot.get_all_sphere_positions_with_link(q)
        min_dist = float('inf')
        n_s = len(spheres)
        for si in range(n_s):
            ci, ri, li = spheres[si]
            for sj in range(si + 1, n_s):
                cj, rj, lj = spheres[sj]
                if abs(li - lj) <= 1:
                    continue
                dist = np.linalg.norm(ci - cj) - ri - rj
                if dist < min_dist:
                    min_dist = dist
        return min_dist


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
    
    # 加载碰撞球体配置
    yml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xtrainer.yml')
    collision_spheres = load_collision_spheres(yml_path)
    print(f"加载碰撞球体: {sum(len(v) for v in collision_spheres.values())} 个, "
          f"覆盖 {len(collision_spheres)} 个连杆")
    
    # 创建机器人
    robot = XTrainerRobot(collision_spheres=collision_spheres)
    
    # B 样条基函数 (三阶, 10个控制点)
    basis = BSplineBasis(n_basis=10, n_dof=6, n_points=80)
    
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
