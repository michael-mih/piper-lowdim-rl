from controllers.controller import Controller
class FSMActor:
    slip_tolerance_N = 0.2
    def __init__(self, controller: Controller):
        self.controller = controller
        self.min = None
        self.min_checkpoint = None

        self.step_count = 0
        self.next_step_interval = 0
        self.three_force_buffer_left = [0.0,0.0,0.0]
        self.three_force_buffer_right = [0.0,0.0,0.0]
        self.grip_delta = -0.6
        


    def tighten(self, increment: float):
        c = list(self.controller.get_joint_angle_cmd())
        lo, _ = self._gripper_bounds()
        c[6] = max(lo, c[6] - 2.0 * increment)
        self.controller.send_joint_angle_cmd(c)

    def loosen(self, increment: float):
        c = list(self.controller.get_joint_angle_cmd())
        _, hi = self._gripper_bounds()
        c[6] = min(hi, c[6] + 2.0 * increment)
        self.controller.send_joint_angle_cmd(c)

    def _gripper_bounds(self):
        command_bounds = getattr(self.controller, "command_bounds", None)
        if command_bounds and 6 in command_bounds:
            lo, hi = command_bounds[6]
            return float(lo), float(hi)

        joint_bounds = getattr(self.controller, "joint_bounds", None)
        if joint_bounds and len(joint_bounds) >= 14:
            return float(joint_bounds[12]), float(joint_bounds[13])
        return 0.0, 0.07

    def is_slip(self) -> bool:
        if self.controller.get_force_left() < 0.01 or self.controller.get_force_right() < 0.01:
            return True
        l = 0
        r = 0
        for i in range(0, 3):
            if self.three_force_buffer_left[i] < 0.2:
                l +=1
            if self.three_force_buffer_right[i] < 0.2:
                r +=1
            if l == 3 or r == 3:
                pass
            if self.three_force_buffer_left[0] - self.three_force_buffer_left[1] < self.grip_delta or self.three_force_buffer_right[0] - self.three_force_buffer_right[1] < self.grip_delta:
                return True
        return False
    
    #1 step:0.01 sec, 100 steps:1sec ?
    def is_converged(self, interval) -> bool:
        interval *= 100
        if self.step_count >= self.next_step_interval:
            if self.min_checkpoint is not None and self.min_checkpoint < self.controller.get_force_average():
                return True
            self.next_step_interval = self.step_count + interval
            self.min_checkpoint = self.min
            return False
    
    def step(self):
        self.step_count += 1
        
        self.three_force_buffer_left.insert(0, self.controller.get_force_left())
        self.three_force_buffer_left = self.three_force_buffer_left[:3]
        self.three_force_buffer_right.insert(0, self.controller.get_force_right())
        self.three_force_buffer_right = self.three_force_buffer_right[:3]
        if self.min is None or self.controller.get_force_average() < self.min:
            self.min = self.controller.get_force_average()
        self.controller.step()
