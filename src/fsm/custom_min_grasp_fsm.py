from controllers.controller import Controller


class FSMActor:
    collapse_drop_threshold_N = 0.6
    collapse_force_ratio = 0.5
    contact_loss_threshold_N = 0.01
    convergence_tolerance_N = 0.01

    def __init__(self, controller: Controller):
        self.controller = controller
        self.min = None
        self.min_checkpoint = None
        self.step_count = 0
        self.next_step_interval = 0
        self.three_force_buffer_left = [0.0, 0.0, 0.0]
        self.three_force_buffer_right = [0.0, 0.0, 0.0]

    def reset_tracking(self):
        left_force = self.controller.get_force_left()
        right_force = self.controller.get_force_right()
        self.three_force_buffer_left = [left_force] * 3
        self.three_force_buffer_right = [right_force] * 3
        self.min = self.controller.get_force_average()
        self.min_checkpoint = None
        self.step_count = 0
        self.next_step_interval = 0


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
        if (
            self.controller.get_force_left() < self.contact_loss_threshold_N
            and self.controller.get_force_right() < self.contact_loss_threshold_N
        ):
            return True

        left_step_drop = (
            self.three_force_buffer_left[1] - self.three_force_buffer_left[0]
        )
        right_step_drop = (
            self.three_force_buffer_right[1] - self.three_force_buffer_right[0]
        )
        left_force_collapsed = (
            self.three_force_buffer_left[0]
            < self.collapse_force_ratio * self.three_force_buffer_left[1]
        )
        right_force_collapsed = (
            self.three_force_buffer_right[0]
            < self.collapse_force_ratio * self.three_force_buffer_right[1]
        )
        return (
            (
                left_step_drop > self.collapse_drop_threshold_N
                and left_force_collapsed
            )
            and (
                right_step_drop > self.collapse_drop_threshold_N
                and right_force_collapsed
            )
        )
    
    #1 step:0.01 sec, 100 steps:1sec ?
    def is_converged(self, interval) -> bool:
        interval = max(1, int(interval * 100))
        if self.step_count >= self.next_step_interval:
            converged = (
                self.min_checkpoint is not None
                and abs(self.min_checkpoint - self.min)
                <= self.convergence_tolerance_N
            )
            self.next_step_interval = self.step_count + interval
            self.min_checkpoint = self.min
            return converged
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
