"""
数值 IK 求解器
支持:
1. Jacobian 伪逆求解
2. LM (Levenberg-Marquardt) 求解
3. 自定义末端约束求解
"""

import numpy as np
from typing import Tuple, Optional, List


class NumericalIK:
    """数值 IK 求解器"""
    
    def __init__(self, robot):
        self.robot = robot
        self.max_iter = 100
        self.tolerance = 1e-6
        
    def solve(self, 
             target_pos: np.ndarray,
             initial_guess: np.ndarray = None,
             method: str = 'lm') -> Tuple[np.ndarray, bool]:
        """
        求解 IK
        
        Args:
            target_pos: 目标末端位置
            initial_guess: 初始关节角猜测
            method: 'pseudo_inv', 'lm', 'damped'
            
        Returns:
            (解, 是否成功)
        """
        if initial_guess is None:
            initial_guess = np.zeros(self.robot.n_dof)
            
        q = initial_guess.copy()
        
        for i in range(self.max_iter):
            # 前向运动学
            pos, _ = self.robot.forward_kinematics(q)
            
            # 误差
            error = target_pos - pos
            
            if np.linalg.norm(error) < self.tolerance:
                return q, True
                
            # 雅可比
            J = self._compute_jacobian(q)
            
            if method == 'pseudo_inv':
                # 伪逆
                q += 0.5 * np.linalg.pinv(J[:3, :]) @ error
                
            elif method == 'damped':
                # 阻尼最小二乘
                damping = 0.1
                JtJ = J[:3, :] @ J[:3, :].T
                q += J[:3, :].T @ np.linalg.solve(JtJ + damping * np.eye(3), error)
                
            elif method == 'lm':
                # Levenberg-Marquardt
                lam = 0.01
                JtJ = J[:3, :] @ J[:3, :].T
                delta = J[:3, :].T @ np.linalg.solve(JtJ + lam * np.eye(3), error)
                q += delta
                
                # 线搜索
                pos_new, _ = self.robot.forward_kinematics(q)
                error_new = target_pos - pos_new
                
                if np.linalg.norm(error_new) < np.linalg.norm(error):
                    lam *= 0.5
                else:
                    lam *= 2
                    q -= delta
                    
        return q, False
    
    def _compute_jacobian(self, q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """数值雅可比"""
        n = self.robot.n_dof
        pos, _ = self.robot.forward_kinematics(q)
        J = np.zeros((6, n))
        
        for i in range(n):
            q_pert = q.copy()
            q_pert[i] += eps
            pos_pert, _ = self.robot.forward_kinematics(q_pert)
            J[:3, i] = (pos_pert - pos) / eps
            
        return J


class TaskSpaceIK:
    """
    任务空间 IK
    FACTO 论文中使用的方法：
    在函数空间优化时，需要将末端位置约束映射回关节空间
    """
    
    def __init__(self, robot):
        self.robot = robot
        self.ik = NumericalIK(robot)
        
    def solve_with_constraints(self,
                             target_pos: np.ndarray,
                             constraints: List[dict] = None,
                             max_iter: int = 50) -> Tuple[np.ndarray, bool]:
        """
        带约束的 IK
        
        constraints: [{'type': 'joint_limit', 'limits': [...]}, 
                     {'type': 'obstacle', 'position': [...], 'radius': ...}]
        """
        q = np.zeros(self.robot.n_dof)
        
        for _ in range(max_iter):
            # 基础 IK
            pos, _ = self.robot.forward_kinematics(q)
            error = target_pos - pos
            
            # 雅可比
            J = self._jacobian(q)
            
            # 零空间处理约束
            if constraints:
                # 投影到约束的零空间
                J_constraint = self._constraint_jacobian(constraints)
                if J_constraint.size > 0:
                    # 零空间投影
                    P = np.eye(self.robot.n_dof) - np.linalg.pinv(J) @ J
                    q += P @ J_constraint.T @ np.linalg.norm(error) * 0.01
            
            # 梯度下降
            q += 0.1 * np.linalg.pinv(J[:3, :]) @ error
            
            if np.linalg.norm(error) < 1e-4:
                return q, True
                
        return q, False
    
    def _jacobian(self, q: np.ndarray) -> np.ndarray:
        return self.ik._compute_jacobian(q)
    
    def _constraint_jacobian(self, constraints: List[dict]) -> np.ndarray:
        """约束雅可比"""
        J = np.zeros((0, self.robot.n_dof))
        
        for c in constraints:
            if c['type'] == 'obstacle':
                # 避障梯度
                pos, _ = self.robot.forward_kinematics(c.get('q', np.zeros(self.robot.n_dof)))
                dist = np.linalg.norm(pos - c['position'])
                if dist < c['radius']:
                    grad = (pos - c['position']) / (dist + 1e-6)
                    J = np.vstack([J, grad])
                    
        return J
    
    def solve_batch(self,
                   positions: List[np.ndarray],
                   method: str = 'lm') -> List[np.ndarray]:
        """
        批量求解 IK (用于轨迹)
        """
        solutions = []
        q = np.zeros(self.robot.n_dof)
        
        for pos in positions:
            q, success = self.ik.solve(pos, q, method)
            solutions.append(q.copy())
            
        return solutions


class FABRIKForward:
    """
    FABRIK (Forward And Backward Reaching Inverse Kinematics)
    是一种简单的启发式 IK 方法
    """
    
    def __init__(self, robot):
        self.robot = robot
        
    def solve(self, 
             target: np.ndarray,
             tolerance: float = 0.01,
             max_iter: int = 10) -> Tuple[np.ndarray, bool]:
        """FABRIK 求解"""
        
        # 简化实现
        q = np.zeros(self.robot.n_dof)
        
        for _ in range(max_iter):
            pos, _ = self.robot.forward_kinematics(q)
            error = target - pos
            
            if np.linalg.norm(error) < tolerance:
                return q, True
                
            # 简单的梯度更新
            J = self._jacobian(q)
            q += np.linalg.pinv(J[:3, :]) @ error * 0.5
            
        return q, False
    
    def _jacobian(self, q: np.ndarray) -> np.ndarray:
        eps = 1e-6
        pos, _ = self.robot.forward_kinematics(q)
        J = np.zeros((6, self.robot.n_dof))
        
        for i in range(self.robot.n_dof):
            q_pert = q.copy()
            q_pert[i] += eps
            pos_pert, _ = self.robot.forward_kinematics(q_pert)
            J[:3, i] = (pos_pert - pos) / eps
            
        return J


# IK 方法选择
def create_ik_solver(robot, method='lm'):
    """工厂函数：创建 IK 求解器"""
    
    if method == 'lm' or method == 'numerical':
        return NumericalIK(robot)
    elif method == 'task_space':
        return TaskSpaceIK(robot)
    elif method == 'fabrik':
        return FABRIKForward(robot)
    else:
        return NumericalIK(robot)
