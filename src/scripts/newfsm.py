from controllers.sim_controller import SimController
from scripts.build_sim import combined_xml
from scripts.sim_box import set_box_mass
from fsm.custom_min_grasp_fsm import FSMActor
import argparse

import glfw  # 用于检查窗口关闭事件
import time

def main():
    parser = argparse.ArgumentParser(description="policy training args")
    parser.add_argument("--model-path", default=str(combined_xml), help="MuJoCo XML path.")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--prep-max-steps", type=int, default=None)
    parser.add_argument("--fsm-max-steps", type=int, default=None)
    parser.add_argument("--reset-steps", type=int, default=2000)
    parser.add_argument(
        "--box-mass",
        "--box-mass-kg",
        dest="box_mass_kg",
        type=float,
        default=None,
        help="Override the simulated grasp-box mass in kilograms.",
    )
    args = parser.parse_args()

    controller = SimController(
        pid_controllers=[],
        model_path=args.model_path,
        render=not args.no_render,
    )
    if args.box_mass_kg is not None:
        set_box_mass(controller, args.box_mass_kg)
        print(f"box_mass_kg={args.box_mass_kg}")
    
    fsmActor = FSMActor(controller=controller)
    target_angles = [0, 1.5, -0.3, 0, -0.7, 0, 0.06]
    controller.set_initial_position(target_angles)
    gripper_delta = 0.0001
    arm_delta = 0.001
    prep_steps = 0
    while (
        target_angles[4] > -1.2
        and (args.prep_max_steps is None or prep_steps < args.prep_max_steps)
    ):
        controller.send_joint_angle_cmd(target_angles)
        controller.step()
        # Start the minimum-force search from a grasp that can support the
        # heaviest configured box, then loosen until a collapse is observed.
        if min(controller.get_force_left(), controller.get_force_right()) > 3.0:
            gripper_delta = 0
            #example joint 5 movement
            #target_angles[4] = -1.2
        if gripper_delta == 0:

            if target_angles[4] > -1.2:
                target_angles[4]-=arm_delta
        #example gripper movement
        target_angles[6] = max(0.0, target_angles[6] - 2.0 * gripper_delta)
        prep_steps += 1
        if _window_should_close(controller):
            break 
        
        time.sleep(0.01)
    initial_config = controller.get_joint_angle_cmd()
    increment = 0.00001
    iteration = 0
    total_iterations = 4
    converge_sum = 0
    stop = False
    fsm_steps = 0
    converged_force = None
    fsmActor.reset_tracking()
    print("iteration 1")
    while (
        iteration < total_iterations
        and (args.fsm_max_steps is None or fsm_steps < args.fsm_max_steps)
    ):
        fsmActor.step()
        fsm_steps += 1
        if fsmActor.is_converged(10):
            # The old FSM measured collapse rather than terminating when the
            # running minimum temporarily plateaued.
            pass
        if not stop:
            print("iteration " + str(iteration+1) + ", force: " + str(fsmActor.controller.get_force_average()))
        if fsmActor.is_slip():
            fsmActor.tighten(increment)
            val = (fsmActor.three_force_buffer_left[1] + fsmActor.three_force_buffer_right[1]) / 2
            print(
                "slipped at "
                f"{val}, current=["
                f"{controller.get_force_left()}, {controller.get_force_right()}], "
                f"left_window={fsmActor.three_force_buffer_left}, "
                f"right_window={fsmActor.three_force_buffer_right}"
            )
            converge_sum += val
            stop = True 
            controller.send_joint_angle_cmd(initial_config)
            i = 0
            while(i < args.reset_steps):
                controller.step()
                i+=1
            fsmActor.reset_tracking()
            iteration += 1
            
            stop = False
            #converge_sum += (fsmActor.three_force_buffer_left[1] + fsmActor.three_force_buffer_right[1]) / 2
            #fsmActor.min = None
            #while(fsmActor.controller.get_force_left() < initial_force or fsmActor.controller.get_force_right() < initial_force):
            #    fsmActor.tighten(increment)
            #    fsmActor.step()
            
            #iteration +=1
            #fsmActor.min = None
        elif not stop:
            fsmActor.loosen(increment)
        #print(controller.get_force_average())
        if _window_should_close(controller):
            break 
        time.sleep(0.01)

    if converged_force is not None:
        print("converged at " + str(converged_force))
    elif iteration > 0:
        print("average pre-slip force " + str(converge_sum / iteration))
    else:
        print("no convergence or slip detected")


def _window_should_close(controller) -> bool:
    viewer = getattr(controller, "viewer", None)
    return bool(viewer is not None and glfw.window_should_close(viewer.window))
    
if __name__ == "__main__":
    main()
