#!/usr/bin/env python3
'''
To test this script, run the following commands in separate terminals:
- ros2 launch ariac_gazebo ariac.launch.py trial_name:=tutorial competitor_pkg:=ariac_tutorials
- ros2 run ariac_tutorials tutorial_9.py
'''

import rclpy
from ariac_tutorials.competition_interface import CompetitionInterface
from ariac_msgs.msg import Part

def main(args=None):
    rclpy.init(args=args)
    interface = CompetitionInterface()
    interface.start_competition()
    
    part_to_pick = Part()

    interface.add_objects_to_planning_scene()
    interface.move_floor_robot_home()
    interface.floor_robot_pick_bin_part(part_to_pick)

    interface.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
