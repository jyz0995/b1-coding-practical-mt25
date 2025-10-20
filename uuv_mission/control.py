"""
PD feedback controller for submarine depth control
"""

class PDController:
    """
    PD Controller: u[t] = KP * e[t] + KD * (e[t] - e[t-1])
    """
    
    def __init__(self, kp: float = 0.15, kd: float = 0.6):
        self.KP = kp
        self.KD = kd
        self.previous_error = 0.0
    
    def compute_control(self, reference: float, output: float) -> float:
        """Calculate control action based on reference and current output"""
        error = reference - output
        control_action = self.KP * error + self.KD * (error - self.previous_error)
        self.previous_error = error
        return control_action
    
    def reset(self):
        """Reset controller state"""
        self.previous_error = 0.0