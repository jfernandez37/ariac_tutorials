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

def main(args=None):
    rclpy.init(args=args)
    interface = CompetitionInterface()
    executor = MultiThreadedExecutor()
    executor.add_node(interface)

    spin_thread = threading.Thread(target=executor.spin)
    spin_thread.start()
    
    interface.move_floor_robot_home()
    interface.move_ceiling_robot_home()

    interface.start_competition()
 

    interface.complete_orders()

    interface.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
