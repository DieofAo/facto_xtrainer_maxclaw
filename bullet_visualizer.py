"""
PyBullet 可视化器 - 播放 FACTO 规划的轨迹

功能：
  1. 加载 XTrainer 机械臂 URDF 模型 (带 mesh)
  2. 调用 FACTO 规划轨迹
  3. 在 PyBullet GUI 中实时播放轨迹动画
  4. 支持障碍物可视化

使用方式：
  conda activate facto_xtrainer
  python bullet_visualizer.py
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import os

from facto_xtrainer import XTrainerRobot, BSplineBasis, FACTOFull, TimeParameterizer


class BulletVisualizer:
    """PyBullet 可视化器"""

    def __init__(self, urdf_path: str = None):
        """
        初始化 PyBullet 环境并加载机械臂

        Args:
            urdf_path: URDF 文件路径，默认为项目根目录下的 xtrainer_arm.urdf
        """
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.urdf_path = urdf_path or os.path.join(self.base_dir, "xtrainer_arm.urdf")

        # 启动 PyBullet GUI
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # 设置相机视角
        p.resetDebugVisualizerCamera(
            cameraDistance=1.0,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.3],
        )

        # 加载地面和机械臂
        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF(
            self.urdf_path,
            basePosition=[0, 0, 0],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
        )

        # 解析可控关节（排除 fixed 关节）
        self.joint_indices = []
        self.joint_names = []
        for i in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, i)
            joint_type = info[2]
            if joint_type == p.JOINT_REVOLUTE:
                self.joint_indices.append(i)
                self.joint_names.append(info[1].decode("utf-8"))

        print(f"[Bullet] 加载 URDF: {self.urdf_path}")
        print(f"[Bullet] 可控关节 ({len(self.joint_indices)}): {self.joint_names}")

        # 障碍物视觉体列表
        self.obstacle_visual_ids = []

    def set_joint_positions(self, q: np.ndarray):
        """设置所有关节角度（无物理仿真，直接 reset）"""
        for idx, joint_id in enumerate(self.joint_indices):
            if idx < len(q):
                p.resetJointState(self.robot_id, joint_id, q[idx])

    def add_obstacle(self, position: np.ndarray, radius: float = 0.05):
        """在场景中添加球形障碍物"""
        visual_id = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=[1.0, 0.5, 0.0, 0.8],
        )
        body_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_id,
            basePosition=position.tolist(),
        )
        self.obstacle_visual_ids.append(body_id)
        return body_id

    def add_trajectory_trail(self, trajectory: np.ndarray, color: list = None):
        """绘制轨迹末端路径线（用 debug line 连接相邻帧的末端位置）"""
        color = color or [0.2, 0.6, 1.0]
        for i in range(len(trajectory) - 1):
            self.set_joint_positions(trajectory[i])
            pos_a = self._get_end_effector_pos()
            self.set_joint_positions(trajectory[i + 1])
            pos_b = self._get_end_effector_pos()
            if pos_a is not None and pos_b is not None:
                p.addUserDebugLine(
                    pos_a, pos_b, lineColorRGB=color, lineWidth=2, lifeTime=0
                )

    def _get_end_effector_pos(self):
        """获取末端执行器在 bullet 世界中的位置"""
        if not self.joint_indices:
            return None
        # 末端 link = 最后一个可控关节对应的 link
        ee_link = self.joint_indices[-1]
        state = p.getLinkState(self.robot_id, ee_link)
        return list(state[0])

    def play_trajectory(self, trajectory: np.ndarray, dt: float = 0.03,
                        loop: bool = False, time_info: dict = None):
        """
        播放轨迹动画（支持变速播放）

        Args:
            trajectory: (N, n_dof) 关节角矩阵
            dt: 等速模式的帧间隔（秒），仅在 time_info=None 时使用
            loop: 是否循环播放
            time_info: TimeParameterizer 输出的时间参数化信息
                       如果提供，则用变速播放（转弯慢、直线快）
        """
        if time_info is not None:
            dt_arr = time_info['dt']
            print(f"[Bullet] 变速播放: {len(trajectory)} 帧, "
                  f"总时间={time_info['total_time']:.3f}s, "
                  f"dt=[{dt_arr.min():.4f}, {dt_arr.max():.4f}]s")
        else:
            dt_arr = None
            print(f"[Bullet] 等速播放: {len(trajectory)} 帧, dt={dt}s")

        # 先画末端轨迹线
        self.add_trajectory_trail(trajectory)

        # 回到起点
        self.set_joint_positions(trajectory[0])
        time.sleep(0.5)

        try:
            while True:
                # 正向播放
                for i, q in enumerate(trajectory):
                    self.set_joint_positions(q)
                    p.stepSimulation()
                    if dt_arr is not None and i < len(dt_arr):
                        time.sleep(dt_arr[i])
                    else:
                        time.sleep(dt)

                # 终点停留
                time.sleep(0.5)

                if not loop:
                    break

                # 反向播放
                n = len(trajectory)
                for i in range(n - 1, -1, -1):
                    self.set_joint_positions(trajectory[i])
                    p.stepSimulation()
                    if dt_arr is not None and i > 0:
                        time.sleep(dt_arr[i - 1])
                    else:
                        time.sleep(dt)

                time.sleep(0.5)

        except KeyboardInterrupt:
            print("\n[Bullet] 用户中断播放")

    def wait_for_exit(self):
        """保持窗口打开，直到用户关闭"""
        print("[Bullet] 按 Ctrl+C 或关闭窗口退出")
        try:
            while p.isConnected(self.physics_client):
                p.stepSimulation()
                time.sleep(1.0 / 240)
        except (KeyboardInterrupt, Exception):
            pass
        finally:
            self.disconnect()

    def disconnect(self):
        """断开 PyBullet 连接"""
        if p.isConnected(self.physics_client):
            p.disconnect()
            print("[Bullet] 已断开连接")


def main():
    """主函数：规划轨迹并用 PyBullet 播放"""

    print("=" * 50)
    print("FACTO + PyBullet 可视化")
    print("=" * 50)

    # ---- 1. FACTO 轨迹规划 ----
    robot = XTrainerRobot()
    basis = BSplineBasis(n_basis=10, n_dof=6, n_points=80)
    facto = FACTOFull(robot, basis)
    timer = TimeParameterizer(robot.joint_velocity_limits, cruise_ratio=0.8)

    start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    goal = np.array([0.5, -1.3, 1.2, 1.1, -2.2, 0.8])

    # 障碍物定义（工作空间坐标，单位: 米）
    obstacles = [
        np.array([0.3, -0.2, 0.25]),
        np.array([-0.15, 0.3, 0.35]),
    ]

    print("\n[FACTO] 规划点到点轨迹...")
    t0 = time.time()
    traj, info = facto.optimize(start, goal, obstacles=obstacles, max_iter=150)
    time_info = timer.parameterize(traj)
    elapsed = time.time() - t0
    print(f"[FACTO] 完成: {info['iterations']} 次迭代, 耗时 {elapsed:.2f}s")
    print(f"[FACTO] 起点误差: {np.linalg.norm(traj[0] - start):.6f} rad")
    print(f"[FACTO] 终点误差: {np.linalg.norm(traj[-1] - goal):.6f} rad")
    print(f"[FACTO] 总运动时间: {time_info['total_time']:.4f}s")
    print(f"[FACTO] 峰值速度比: {time_info['max_speed_ratio']:.2%}")
    print(f"[FACTO] dt范围: [{time_info['dt'].min():.5f}, {time_info['dt'].max():.5f}]s")

    # ---- 2. PyBullet 可视化 ----
    viz = BulletVisualizer()

    # 添加障碍物到场景
    for obs in obstacles:
        viz.add_obstacle(obs, radius=0.05)
        print(f"[Bullet] 添加障碍物: {obs}")

    # 设置起始姿态，预览
    viz.set_joint_positions(start)
    time.sleep(1.0)

    # 变速播放轨迹（循环播放，Ctrl+C 停止）
    viz.play_trajectory(traj, loop=True, time_info=time_info)

    # 保持窗口
    viz.wait_for_exit()


if __name__ == "__main__":
    main()
