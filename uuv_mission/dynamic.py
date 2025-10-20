from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .terrain import generate_reference_and_limits
from .control import PDController

class Submarine:
    def __init__(self):

        self.mass = 1
        self.drag = 0.1
        self.actuator_gain = 1

        self.dt = 1 # Time step for discrete time simulation

        self.pos_x = 0
        self.pos_y = 0
        self.vel_x = 1 # Constant velocity in x direction
        self.vel_y = 0


    def transition(self, action: float, disturbance: float):
        self.pos_x += self.vel_x * self.dt
        self.pos_y += self.vel_y * self.dt

        force_y = -self.drag * self.vel_y + self.actuator_gain * (action + disturbance)
        acc_y = force_y / self.mass
        self.vel_y += acc_y * self.dt

    def get_depth(self) -> float:
        return self.pos_y
    
    def get_position(self) -> tuple:
        return self.pos_x, self.pos_y
    
    def reset_state(self):
        self.pos_x = 0
        self.pos_y = 0
        self.vel_x = 1
        self.vel_y = 0
    
class Trajectory:
    def __init__(self, position: np.ndarray):
        self.position = position  
        
    def plot(self):
        plt.plot(self.position[:, 0], self.position[:, 1])
        plt.show()

    def plot_completed_mission(self, mission: Mission):
        x_values = np.arange(len(mission.reference))
        min_depth = np.min(mission.cave_depth)
        max_height = np.max(mission.cave_height)

        plt.fill_between(x_values, mission.cave_height, mission.cave_depth, color='blue', alpha=0.3)
        plt.fill_between(x_values, mission.cave_depth, min_depth*np.ones(len(x_values)), 
                         color='saddlebrown', alpha=0.3)
        plt.fill_between(x_values, max_height*np.ones(len(x_values)), mission.cave_height, 
                         color='saddlebrown', alpha=0.3)
        plt.plot(self.position[:, 0], self.position[:, 1], label='Trajectory')
        plt.plot(mission.reference, 'r', linestyle='--', label='Reference')
        plt.legend(loc='upper right')
        plt.show()

@dataclass
class Mission:
    reference: np.ndarray
    cave_height: np.ndarray
    cave_depth: np.ndarray

    @classmethod
    def random_mission(cls, duration: int, scale: float):
        (reference, cave_height, cave_depth) = generate_reference_and_limits(duration, scale)
        return cls(reference, cave_height, cave_depth)

    @classmethod
    def from_csv(cls, file_name: str):
        # 读取 CSV 文件
        data = pd.read_csv(file_name)
        
        # 将每一列转换为 numpy 数组
        reference = data['reference'].to_numpy()
        cave_height = data['cave_height'].to_numpy()
        cave_depth = data['cave_depth'].to_numpy()
        
        # 返回 Mission 实例
        return cls(reference, cave_height, cave_depth)

class ClosedLoop:
    
    def __init__(self, submarine: Submarine, kp: float = 0.15, kd: float = 0.6):
        """
        Initialize closed-loop system
        
        Args:
            submarine: Submarine instance
            kp: proportional gain
            kd: derivative gain
        """
        self.submarine = submarine
        self.controller = PDController(kp, kd)
        self.dt = submarine.dt
    
    def simulate(self, mission: Mission, disturbances: np.ndarray = None) -> Trajectory:
        """Simulate closed-loop system with optional disturbances"""
        self.submarine.reset_state()
        self.controller.reset()
        positions = []
        
        # If no disturbances provided, use zeros
        if disturbances is None:
            disturbances = np.zeros(len(mission.reference))
        
        for t in range(len(mission.reference)):
            current_depth = self.submarine.get_depth()
            reference_depth = mission.reference[t]
            
            # Compute control action
            u = self.controller.compute_control(reference_depth, current_depth)
            
            # Apply transition with disturbance
            self.submarine.transition(u, disturbances[t])
            
            # Record position
            positions.append(self.submarine.get_position())
        
        return Trajectory(np.array(positions))
        
    def simulate_with_random_disturbances(self, mission: Mission, variance: float = 0.5) -> Trajectory:
        disturbances = np.random.normal(0, variance, len(mission.reference))
        return self.simulate(mission, disturbances)
