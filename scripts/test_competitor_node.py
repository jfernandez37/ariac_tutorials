#!/usr/bin/env python3
'''
To test this script, run the following commands in separate terminals:
- ros2 launch ariac_gazebo ariac.launch.py trial_name:=tutorial competitor_pkg:=ariac_tutorials
- ros2 run ariac_tutorials tutorial_10.py
'''
import threading
import rclpy
from ariac_tutorials.competition_interface import CompetitionInterface
from ariac_msgs.msg import Part
from rclpy.executors import MultiThreadedExecutor
from ariac_tutorials.utils import (
    multiply_pose,
    rpy_from_quaternion,
    rad_to_deg_str,
    quaternion_from_euler,
    build_pose,
    AdvancedLogicalCameraImage,
    Order,
    KittingTask,
    CombinedTask,
    AssemblyTask,
    KittingPart
)

def main(args=None):
    rclpy.init(args=args)
    interface = CompetitionInterface()
    executor = MultiThreadedExecutor()
    executor.add_node(interface)

    spin_thread = threading.Thread(target=executor.spin)
    spin_thread.start()
    
    

    interface.start_competition() 
    __import__("time").sleep(2.0)
    
    interface.move_floor_robot_home()
    interface.move_ceiling_robot_home()
    
    interface.complete_orders()
    # part = Part()
    # part.type = Part.BATTERY
    # part.color = Part.BLUE

    # interface.floor_robot_pick_conveyor_part(part)

    interface.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
