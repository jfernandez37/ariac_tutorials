from argparse import _MutuallyExclusiveGroup
from distutils.command import build
from time import sleep
from math import cos, sin, pi
from copy import copy, deepcopy
import time
import threading
from functools import partial
import PyKDL
from sympy import Quaternion
from ament_index_python import get_package_share_directory
from moveit.planning import MoveItPy
import rclpy
import pyassimp
import yaml
from rclpy.time import Duration, Time
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.parameter import Parameter
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup

from geometry_msgs.msg import PoseStamped, Pose, Point, TransformStamped
from shape_msgs.msg import Mesh, MeshTriangle
from moveit_msgs.msg import CollisionObject, AttachedCollisionObject, PlanningScene
from moveit_msgs.msg import RobotTrajectory as RobotTrajectoryMsg
from trajectory_msgs.msg import JointTrajectoryPoint
from std_msgs.msg import Header

from moveit.core.robot_trajectory import RobotTrajectory
from moveit.core.robot_state import RobotState, robotStateToRobotStateMsg
from moveit_msgs.srv import GetCartesianPath, GetPositionFK, ApplyPlanningScene, GetPlanningScene
from moveit.core.kinematic_constraints import construct_joint_constraint

from ariac_msgs.msg import (
    CompetitionState as CompetitionStateMsg,
    AdvancedLogicalCameraImage as AdvancedLogicalCameraImageMsg,
    Part as PartMsg,
    PartPose as PartPoseMsg,
    Order as OrderMsg,
    AssemblyPart as AssemblyPartMsg,
    AGVStatus as AGVStatusMsg,
    AssemblyTask as AssemblyTaskMsg,
    AssemblyState as AssemblyStateMsg,
    CombinedTask as CombinedTaskMsg,
    VacuumGripperState,
    ConveyorParts,
    BreakBeamStatus
)

from ariac_msgs.srv import (
    MoveAGV,
    VacuumGripperControl,
    ChangeGripper,
    SubmitOrder,
    GetPreAssemblyPoses,
)

from std_srvs.srv import Trigger

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

from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster


class Error(Exception):
  def __init__(self, value: str):
      self.value = value

  def __str__(self):
      return repr(self.value)

class ConveyorPart():
    def __init__(self, part, pose, detection_time):
        self.pose = Pose()
        self.pose = pose

        self.part : PartMsg
        self.part = part

        self.detection_time = detection_time
  
class CompetitionInterface(Node):
    '''
    Class for a competition interface node.

    Args:
        Node (rclpy.node.Node): Parent class for ROS nodes

    Raises:
        KeyboardInterrupt: Exception raised when the user uses Ctrl+C to kill a process
    '''
    _competition_states = {
        CompetitionStateMsg.IDLE: 'idle',
        CompetitionStateMsg.READY: 'ready',
        CompetitionStateMsg.STARTED: 'started',
        CompetitionStateMsg.ORDER_ANNOUNCEMENTS_DONE: 'order_announcements_done',
        CompetitionStateMsg.ENDED: 'ended',
    }
    '''Dictionary for converting CompetitionState constants to strings'''

    _part_colors = {
        PartMsg.RED: 'red',
        PartMsg.BLUE: 'blue',
        PartMsg.GREEN: 'green',
        PartMsg.ORANGE: 'orange',
        PartMsg.PURPLE: 'purple',
    }
    '''Dictionary for converting Part color constants to strings'''

    _part_colors_emoji = {
        PartMsg.RED: '🟥',
        PartMsg.BLUE: '🟦',
        PartMsg.GREEN: '🟩',
        PartMsg.ORANGE: '🟧',
        PartMsg.PURPLE: '🟪',
    }
    '''Dictionary for converting Part color constants to emojis'''

    _part_types = {
        PartMsg.BATTERY: 'battery',
        PartMsg.PUMP: 'pump',
        PartMsg.REGULATOR: 'regulator',
        PartMsg.SENSOR: 'sensor',
    }
    '''Dictionary for converting Part type constants to strings'''

    _destinations = {
        AGVStatusMsg.KITTING: 'kitting station',
        AGVStatusMsg.ASSEMBLY_FRONT: 'front assembly station',
        AGVStatusMsg.ASSEMBLY_BACK: 'back assembly station',
        AGVStatusMsg.WAREHOUSE: 'warehouse',
    }
    '''Dictionary for converting AGVDestination constants to strings'''

    _stations = {
        AssemblyTaskMsg.AS1: 'assembly station 1',
        AssemblyTaskMsg.AS2: 'assembly station 2',
        AssemblyTaskMsg.AS3: 'assembly station 3',
        AssemblyTaskMsg.AS4: 'assembly station 4',
    }
    '''Dictionary for converting AssemblyTask constants to strings'''
    
    _gripper_states = {
        True: 'enabled',
        False: 'disabled'
    }
    '''Dictionary for converting VacuumGripperState constants to strings'''

    _part_heights = {PartMsg.BATTERY : 0.04,
                     PartMsg.PUMP : 0.12,
                     PartMsg.REGULATOR : 0.07,
                     PartMsg.SENSOR : 0.07}
    '''Dictionary for the heights of each part'''

    _quad_offsets = {1 : (-0.08, 0.12),
                     2 : (0.08, 0.12),
                     3 : (-0.08, -0.12),
                     4 : (0.08, -0.12)}

    _rail_positions = {"agv1":-4.5,
                       "agv2":-1.2,
                       "agv3":1.2,
                       "agv4":4.5,
                       "left_bins":3,
                       "right_bins":-3}

    def __init__(self):
        super().__init__('competition_interface')

        sim_time = Parameter(
            "use_sim_time",
            rclpy.Parameter.Type.BOOL,
            True
        )

        self.set_parameters([sim_time])
        
        # ROS2 callback groups
        self.sensor_cb_group = MutuallyExclusiveCallbackGroup()
        self.ariac_cb_group = MutuallyExclusiveCallbackGroup()
        self.moveit_cb_group = MutuallyExclusiveCallbackGroup()
        self.orders_cb_group = ReentrantCallbackGroup()

        # Service client for starting and ending the competition
        self._start_competition_client = self.create_client(Trigger, '/ariac/start_competition')
        self._end_competition_client = self.create_client(Trigger, "/ariac/end_competition")

        # Subscriber to the competition state topic
        self._competition_state_sub = self.create_subscription(
            CompetitionStateMsg,
            '/ariac/competition_state',
            self._competition_state_cb,
            10,
            callback_group=self.ariac_cb_group)
        
        # Store the state of the competition
        self._competition_state: CompetitionStateMsg = None
        
        # Store the number of parts that crossed the beam
        self._conveyor_part_count = 0
        
        # Store whether the beam is broken
        self._object_detected = False

        # Store each camera image as an AdvancedLogicalCameraImage object
        self._camera_image: AdvancedLogicalCameraImage = None

        # Subscriber to the order topic
        self.orders_sub = self.create_subscription(
            OrderMsg,
            '/ariac/orders',
            self._orders_cb,
            10,
            callback_group=self.orders_cb_group)
        
        # Flag for parsing incoming orders
        self._parse_incoming_order = True
        
        # List of orders
        self._orders = []
        
        # Subscriber to the floor gripper state topic
        self._floor_robot_gripper_state_sub = self.create_subscription(
            VacuumGripperState,
            '/ariac/floor_robot_gripper_state',
            self._floor_robot_gripper_state_cb,
            qos_profile_sensor_data,
            callback_group=self.ariac_cb_group)

        # Service client for turning on/off the vacuum gripper on the floor robot
        self._floor_gripper_enable = self.create_client(
            VacuumGripperControl,
            "/ariac/floor_robot_enable_gripper")
        
        self._floor_2_gripper_enable = self.create_client(
            VacuumGripperControl,
            "/ariac/floor_robot_2_enable_gripper"
        )

        # Attribute to store the current state of the floor robot gripper
        self._floor_robot_gripper_state = VacuumGripperState()

        # Attribute to store the current state of the floor robot 2 gripper
        self._floor_robot_2_gripper_state = VacuumGripperState()

        # Moveit_py variables
        self._ariac_robots = MoveItPy(node_name="ariac_robots_moveit_py")
        self._ariac_robots_state = RobotState(self._ariac_robots.get_robot_model())

        self._floor_robot = self._ariac_robots.get_planning_component("floor_robot")
        # self._floor_robot_2 = self._ariac_robots.get_planning_component("floor_robot_2")

        self._planning_scene_monitor = self._ariac_robots.get_planning_scene_monitor()

        self._world_collision_objects = []

        # Parts found in the bins
        self.bin_1_parts = []
        self.bin_2_parts = []
        self.bin_1_camera_pose = Pose()
        self.bin_2_camera_pose = Pose()

        # Tray information
        self._kts1_trays = []
        self._kts2_trays = []
        self._kts1_camera_pose = Pose()
        self._kts2_camera_pose = Pose()

        # service clients
        self.get_cartesian_path_client = self.create_client(GetCartesianPath, "compute_cartesian_path")
        self.get_position_fk_client = self.create_client(GetPositionFK, "compute_fk")

        # Camera subs
        self.left_bins_camera_sub = self.create_subscription(AdvancedLogicalCameraImageMsg,
                                                             "/ariac/sensors/bin_1_camera/image",
                                                             self.bin_1_camera_cb,
                                                             qos_profile_sensor_data,
                                                             callback_group=self.sensor_cb_group)
        self.right_bins_camera_sub = self.create_subscription(AdvancedLogicalCameraImageMsg,
                                                             "/ariac/sensors/bin_12_camera/image",
                                                             self.bin_2_camera_cb,
                                                             qos_profile_sensor_data,
                                                             callback_group=self.sensor_cb_group)
        
        
        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.tf_broadcaster = StaticTransformBroadcaster(self)
        self.static_transforms = []

        self.floor_robot_attached_part_ = PartMsg()
        self.floor_robot_2_attached_part_ = PartMsg()

        self._change_gripper_client = self.create_client(ChangeGripper, "/ariac/floor_robot_change_gripper")
        
        # Planning Scene Info
        self.planning_scene_sub = self.create_subscription(PlanningScene,
                                                           "/planning_scene",
                                                            self.get_planning_scene_msg,
                                                            10,
                                                            callback_group=self.moveit_cb_group)
        self.planning_scene_msg = PlanningScene()

        # Meshes file path
        self.mesh_file_path = get_package_share_directory("ariac_tutorials") + "/meshes/"
        
        self.floor_joint_positions_arrs = {
            "floor_kts1_js_":[4.0,1.57,-1.57,1.57,-1.57,-1.57,0.0],
            "floor_kts2_js_":[-4.0,-1.57,-1.57,1.57,-1.57,-1.57,0.0],
            "floor_robot_home":[0.0, 0.0, -1.57, 1.57, -1.571, -1.571, 0.0],
            "floor_robot_2_home":[0.0, 0.0, -1.57, 1.57, -1.571, -1.571, 0.0],
            "floor_conveyor_js_":[0.0,3.14,-0.9162979,2.04204, -2.67035, -1.57, 0.0]
        }
        for i in range(1,5):
            self.floor_joint_positions_arrs[f"agv{i}"]=[self._rail_positions[f"agv{i}"],0.0,-1.57,1.57,-1.57,-1.57,0.0]
            
        self.floor_position_dict = {key:self._create_floor_joint_position_state(self.floor_joint_positions_arrs[key])
                                      for key in self.floor_joint_positions_arrs.keys()}
        
        self.current_order = None

        # Conveyor belt
        self.conveyor_parts_expected = []
        self.conveyor_parts_sub = self.create_subscription(ConveyorParts,
                                                           '/ariac/conveyor_parts',
                                                           self.conveyor_parts_cb,
                                                           qos_profile_sensor_data,
                                                           callback_group=self.ariac_cb_group)
        self.conveyor_camera_sub = self.create_subscription(AdvancedLogicalCameraImageMsg,
                                                            '/ariac/sensors/conveyor_camera/image',
                                                            self.conveyor_camera_cb,
                                                            qos_profile_sensor_data,
                                                            callback_group=self.sensor_cb_group)
        self.conveyor_parts_detected = []
        self.conveyor_camera_pose = Pose()
        self.conveyor_camera_image = None
        self.part_already_scanned = False
        self.last_conveyor_image_time = None
        self.part_count = 0

        self.breakbeam_sub_ = self.create_subscription(BreakBeamStatus,
                                                       '/ariac/sensors/conveyor_breakbeam/change',
                                                       self.breakbeam_cb,
                                                       qos_profile_sensor_data,
                                                       callback_group=self.sensor_cb_group)
        self.breakbeam_recieved_data = False
        self.breakbeam_pose = Pose()
        self.breakbeam_status = False
        self.conveyor_parts = []
        self.parts_on_conveyor = []

        self.conveyor_speed = 0.2


    @property
    def orders(self):
        return self._orders

    @property
    def camera_image(self):
        return self._camera_image

    @property
    def conveyor_part_count(self):
        return self._conveyor_part_count

    @property
    def parse_incoming_order(self):
        return self._parse_incoming_order

    @parse_incoming_order.setter
    def parse_incoming_order(self, value):
        self._parse_incoming_order = value

    def _orders_cb(self, msg: OrderMsg):
        '''Callback for the topic /ariac/orders
        Arguments:
            msg -- Order message
        '''
        order = Order(msg)
        self._orders.append(order)
        if self._parse_incoming_order:
            self.get_logger().info(self._parse_order(order))

    def _competition_state_cb(self, msg: CompetitionStateMsg):
        '''Callback for the topic /ariac/competition_state
        Arguments:
            msg -- CompetitionState message
        '''
        # Log if competition state has changed
        if self._competition_state != msg.competition_state:
            state = CompetitionInterface._competition_states[msg.competition_state]
            self.get_logger().info(f'Competition state is: {state}', throttle_duration_sec=1.0)
        
        self._competition_state = msg.competition_state
        
    def _floor_robot_gripper_state_cb(self, msg: VacuumGripperState):
        '''Callback for the topic /ariac/floor_robot_gripper_state

        Arguments:
            msg -- VacuumGripperState message
        '''
        self._floor_robot_gripper_state = msg
    

    def start_competition(self):
        '''Function to start the competition.
        '''
        self.get_logger().info('Waiting for competition to be ready')

        if self._competition_state == CompetitionStateMsg.STARTED:
            return
        # Wait for competition to be ready
        while self._competition_state != CompetitionStateMsg.READY:
            pass

        self.get_logger().info('Competition is ready. Starting...')

        # Check if service is available
        if not self._start_competition_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error('Service \'/ariac/start_competition\' is not available.')
            return False

        # Create trigger request and call starter service
        request = Trigger.Request()
        future = self._start_competition_client.call_async(request)

        while not future.done():
            pass

        if future.result().success:
            self.get_logger().info('Started competition.')
            return True
        else:
            self.get_logger().warn('Unable to start competition')
            return False
    
    def end_competition(self):
        self.get_logger().info('Ending competition')

        # Check if service is available
        if not self._end_competition_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error('Service \'/ariac/end_competition\' is not available.')
            return

        # Create trigger request and call starter service
        request = Trigger.Request()
        future = self._end_competition_client.call_async(request)

        while not future.done():
            pass

        if future.result().success:
            self.get_logger().info('Ended competition.')
        else:
            self.get_logger().warn('Unable to end competition')

    def parse_advanced_camera_image(self, image: AdvancedLogicalCameraImage) -> str:
        '''
        Parse an AdvancedLogicalCameraImage message and return a string representation.
        '''
        
        if len(image._part_poses) == 0:
            return 'No parts detected'

        output = '\n\n'
        for i, part_pose in enumerate(image._part_poses):
            part_pose: PartPoseMsg
            output += '==========================\n'
            part_color = CompetitionInterface._part_colors[part_pose.part.color].capitalize()
            part_color_emoji = CompetitionInterface._part_colors_emoji[part_pose.part.color]
            part_type = CompetitionInterface._part_types[part_pose.part.type].capitalize()
            output += f'Part {i+1}: {part_color_emoji} {part_color} {part_type}\n'
            output += '--------------------------\n'
            output += 'Camera Frame\n'
            output += '--------------------------\n'
            
            output += '  Position:\n'
            output += f'    x: {part_pose.pose.position.x:.3f} (m)\n'
            output += f'    y: {part_pose.pose.position.y:.3f} (m)\n'
            output += f'    z: {part_pose.pose.position.z:.3f} (m)\n'

            roll, pitch, yaw = rpy_from_quaternion(part_pose.pose.orientation)
            output += '  Orientation:\n'
            output += f'    roll: {rad_to_deg_str(roll)}\n'
            output += f'    pitch: {rad_to_deg_str(pitch)}\n'
            output += f'    yaw: {rad_to_deg_str(yaw)}\n'
            
            part_world_pose = multiply_pose(image._sensor_pose, part_pose.pose)
            output += '--------------------------\n'
            output += 'World Frame\n'
            output += '--------------------------\n'

            output += '  Position:\n'
            output += f'    x: {part_world_pose.position.x:.3f} (m)\n'
            output += f'    y: {part_world_pose.position.y:.3f} (m)\n'
            output += f'    z: {part_world_pose.position.z:.3f} (m)\n'

            roll, pitch, yaw = rpy_from_quaternion(part_world_pose.orientation)
            output += '  Orientation:\n'
            output += f'    roll: {rad_to_deg_str(roll)}\n'
            output += f'    pitch: {rad_to_deg_str(pitch)}\n'
            output += f'    yaw: {rad_to_deg_str(yaw)}\n'

            output += '==========================\n\n'

        return output
    
    def _parse_kitting_task(self, kitting_task: KittingTask):
        '''
        Parses a KittingTask object and returns a string representation.
        Args:
            kitting_task (KittingTask): KittingTask object to parse
        Returns:
            str: String representation of the KittingTask object
        '''
        output = 'Type: Kitting\n'
        output += '==========================\n'
        output += f'AGV: {kitting_task.agv_number}\n'
        output += f'Destination: {CompetitionInterface._destinations[kitting_task.destination]}\n'
        output += f'Tray ID: {kitting_task.tray_id}\n'
        output += 'Products:\n'
        output += '==========================\n'

        quadrants = {1: "Quadrant 1: -",
                     2: "Quadrant 2: -",
                     3: "Quadrant 3: -",
                     4: "Quadrant 4: -"}

        for i in range(1, 5):
            product: KittingPart
            for product in kitting_task.parts:
                if i == product.quadrant:
                    part_color = CompetitionInterface._part_colors[product.part.color].capitalize()
                    part_color_emoji = CompetitionInterface._part_colors_emoji[product.part.color]
                    part_type = CompetitionInterface._part_types[product.part.type].capitalize()
                    quadrants[i] = f'Quadrant {i}: {part_color_emoji} {part_color} {part_type}'
        output += f'\t{quadrants[1]}\n'
        output += f'\t{quadrants[2]}\n'
        output += f'\t{quadrants[3]}\n'
        output += f'\t{quadrants[4]}\n'

        return output

    def _parse_assembly_task(self, assembly_task: AssemblyTask):
        '''
        Parses an AssemblyTask object and returns a string representation.

        Args:
            assembly_task (AssemblyTask): AssemblyTask object to parse

        Returns:
            str: String representation of the AssemblyTask object
        '''
        output = 'Type: Assembly\n'
        output += '==========================\n'
        if len(assembly_task.agv_numbers) == 1:
            output += f'AGV: {assembly_task.agv_number[0]}\n'
        elif len(assembly_task.agv_numbers) == 2:
            output += f'AGV(s): [{assembly_task.agv_numbers[0]}, {assembly_task.agv_numbers[1]}]\n'
        output += f'Station: {self._stations[assembly_task.station].title()}\n'
        output += 'Products:\n'
        output += '==========================\n'

        product: AssemblyPartMsg
        for product in assembly_task.parts:
            part_color = CompetitionInterface._part_colors[product.part.color].capitalize()
            part_color_emoji = CompetitionInterface._part_colors_emoji[product.part.color]
            part_type = CompetitionInterface._part_types[product.part.type].capitalize()

            output += f'Part: {part_color_emoji} {part_color} {part_type}\n'

            output += '  Position:\n'
            output += f'    x: {product.assembled_pose.pose.position.x:.3f} (m)\n'
            output += f'    y: {product.assembled_pose.pose.position.y:.3f} (m)\n'
            output += f'    z: {product.assembled_pose.pose.position.z:.3f} (m)\n'

            roll, pitch, yaw = rpy_from_quaternion(product.assembled_pose.pose.orientation)
            output += '  Orientation:\n'
            output += f'    roll: {rad_to_deg_str(roll)}\n'
            output += f'    pitch: {rad_to_deg_str(pitch)}\n'
            output += f'    yaw: {rad_to_deg_str(yaw)}\n'

            output += f'  Install direction:\n'
            output += f'    x: {product.install_direction.x:.1f}\n'
            output += f'    y: {product.install_direction.y:.1f}\n'
            output += f'    z: {product.install_direction.z:.1f}\n'

        return output

    def _parse_combined_task(self, combined_task: CombinedTask):
        '''
        Parses a CombinedTask object and returns a string representation.

        Args:
            combined_task (CombinedTask): CombinedTask object to parse

        Returns:
            str: String representation of the CombinedTask object
        '''

        output = 'Type: Combined\n'
        output += '==========================\n'
        output += f'Station: {self._stations[combined_task.station].title()}\n'
        output += 'Products:\n'
        output += '==========================\n'

        product: AssemblyPartMsg
        for product in combined_task.parts:
            part_color = CompetitionInterface._part_colors[product.part.color].capitalize()
            part_color_emoji = CompetitionInterface._part_colors_emoji[product.part.color]
            part_type = CompetitionInterface._part_types[product.part.type].capitalize()

            output += f'Part: {part_color_emoji} {part_color} {part_type}\n'

            output += '  Position:\n'
            output += f'    x: {product.assembled_pose.pose.position.x:.3f} (m)\n'
            output += f'    y: {product.assembled_pose.pose.position.y:.3f} (m)\n'
            output += f'    z: {product.assembled_pose.pose.position.z:.3f} (m)\n'

            roll, pitch, yaw = rpy_from_quaternion(product.assembled_pose.pose.orientation)
            output += '  Orientation:\n'
            output += f'    roll: {rad_to_deg_str(roll)}\n'
            output += f'    pitch: {rad_to_deg_str(pitch)}\n'
            output += f'    yaw: {rad_to_deg_str(yaw)}\n'

            output += f'  Install direction:\n'
            output += f'    x: {product.install_direction.x:.1f}\n'
            output += f'    y: {product.install_direction.y:.1f}\n'
            output += f'    z: {product.install_direction.z:.1f}\n'

        return output

    def _parse_order(self, order: Order):
        '''Parse an order message and return a string representation.
        Args:
            order (Order) -- Order message
        Returns:
            String representation of the order message
        '''
        output = '\n\n==========================\n'
        output += f'Received Order: {order.order_id}\n'
        output += f'Priority: {order.order_priority}\n'

        if order.order_type == OrderMsg.KITTING:
            output += self._parse_kitting_task(order.order_task)
        elif order.order_type == OrderMsg.ASSEMBLY:
            output += self._parse_assembly_task(order.order_task)
        elif order.order_type == OrderMsg.COMBINED:
            output += self._parse_combined_task(order.order_task)
        else:
            output += 'Type: Unknown\n'
        return output

    def set_floor_robot_gripper_state(self, state, robot):
        '''Set the gripper state of the floor robot.

        Arguments:
            state -- True to enable, False to disable

        Raises:
            KeyboardInterrupt: Exception raised when the user presses Ctrl+C
        '''
        if self._floor_robot_gripper_state.enabled == state:
            self.get_logger().warn(f'Gripper is already {self._gripper_states[state]}')
            return True

        request = VacuumGripperControl.Request()
        request.enable = state

        if robot == "floor_robot":
            future = self._floor_gripper_enable.call_async(request)
        else:
            future = self._floor_2_gripper_enable.call_async(request)

        while not future.done():
            pass

        if future.result().success:
            self.get_logger().info(f'Changed gripper state to {self._gripper_states[state]}')
            return True
        else:
            self.get_logger().warn('Unable to change gripper state')
            return False
    
    def _call_get_cartesian_path(self, waypoints : list, 
                                  max_velocity_scaling_factor : float, 
                                  max_acceleration_scaling_factor : float,
                                  avoid_collision : bool,
                                  robot : str):

        self.get_logger().info("Getting cartesian path")

        request = GetCartesianPath.Request()

        header = Header()
        header.frame_id = "world"
        header.stamp = self.get_clock().now().to_msg()

        request.header = header
        with self._planning_scene_monitor.read_write() as scene:
            request.start_state = robotStateToRobotStateMsg(scene.current_state)

        if robot == "floor_robot":
            request.group_name = "floor_robot"
            request.link_name = "floor_gripper"
        else:
            request.group_name = "floor_robot_2"
            request.link_name = "floor_2_gripper"
        
        request.waypoints = waypoints
        request.max_step = 0.1
        request.avoid_collisions = avoid_collision
        request.max_velocity_scaling_factor = max_velocity_scaling_factor
        request.max_acceleration_scaling_factor = max_acceleration_scaling_factor

        future = self.get_cartesian_path_client.call_async(request)

        while not future.done():
            pass

        result: GetCartesianPath.Response
        result = future.result()

        if result.fraction < 0.9:
            self.get_logger().error("Unable to plan cartesian trajectory")

        return result.solution
    
    def _plan_and_execute(
        self,
        robot,
        planning_component,
        logger,
        robot_type,
        single_plan_parameters=None,
        multi_plan_parameters=None,
    ):
        """Helper function to plan and execute a motion."""
        # plan to goal
        logger.info("Planning trajectory")
        if multi_plan_parameters is not None:
            plan_result = planning_component.plan(
                multi_plan_parameters=multi_plan_parameters
            )
        elif single_plan_parameters is not None:
            plan_result = planning_component.plan(
                single_plan_parameters=single_plan_parameters
            )
        else:
            plan_result = planning_component.plan()
        # execute the plan
        if plan_result:
            logger.info("Executing plan")
            with self._planning_scene_monitor.read_write() as scene:
                scene.current_state.update(True)
                self._ariac_robots_state = scene.current_state
                robot_trajectory = plan_result.trajectory
            robot.execute(robot_trajectory, controllers=["floor_robot_controller","linear_rail_controller"] if robot_type=="floor_robot" else ["floor_robot_2_controller"])
        else:
            logger.error("Planning failed")
            return False
        return True

    def move_floor_robot_home(self):
        with self._planning_scene_monitor.read_write() as scene:
            self._floor_robot.set_start_state(robot_state = scene.current_state)
            self._floor_robot.set_goal_state(configuration_name="home")
        self._plan_and_execute(self._ariac_robots,self._floor_robot, self.get_logger(),"floor_robot")
        with self._planning_scene_monitor.read_write() as scene:
            scene.current_state.update()
            self._ariac_robots_state = scene.current_state

    def _move_floor_robot_cartesian(self, waypoints, velocity, acceleration, avoid_collision = True):
        trajectory_msg = self._call_get_cartesian_path(waypoints, velocity, acceleration, avoid_collision, "floor_robot")
        with self._planning_scene_monitor.read_write() as scene:

            trajectory = RobotTrajectory(self._ariac_robots.get_robot_model())
            trajectory.set_robot_trajectory_msg(scene.current_state, trajectory_msg)
            trajectory.joint_model_group_name = "floor_robot"

            trajectory_msg: RobotTrajectoryMsg
            point : JointTrajectoryPoint
            point = trajectory_msg.joint_trajectory.points[-1]
            dur = Duration(seconds=point.time_from_start.sec, nanoseconds=point.time_from_start.nanosec)

            self.get_logger().info(f"Motion will take {dur.nanoseconds} nanoseconds to complete")

            scene.current_state.update(True)
            self._ariac_robots_state = scene.current_state

        self.async_execute(trajectory,[])
    
    def _move_floor_robot_2_cartesian(self, waypoints, velocity, acceleration, avoid_collision = True):
        trajectory_msg = self._call_get_cartesian_path(waypoints, velocity, acceleration, avoid_collision, "floor_robot_2")
        with self._planning_scene_monitor.read_write() as scene:

            trajectory = RobotTrajectory(self._ariac_robots.get_robot_model())
            trajectory.set_robot_trajectory_msg(scene.current_state, trajectory_msg)
            trajectory.joint_model_group_name = "floor_robot_2"

            trajectory_msg: RobotTrajectoryMsg
            point : JointTrajectoryPoint
            point = trajectory_msg.joint_trajectory.points[-1]
            dur = Duration(seconds=point.time_from_start.sec, nanoseconds=point.time_from_start.nanosec)

            self.get_logger().info(f"Motion will take {dur.nanoseconds} nanoseconds to complete")

            scene.current_state.update(True)
            self._ariac_robots_state = scene.current_state

        self.async_execute(trajectory,[])

    def _move_floor_robot_to_pose(self,pose : Pose):
        with self._planning_scene_monitor.read_write() as scene:
            self._floor_robot.set_start_state(robot_state = scene.current_state)

            pose_goal = PoseStamped()
            pose_goal.header.frame_id = "world"
            pose_goal.pose = pose
            self._floor_robot.set_goal_state(pose_stamped_msg=pose_goal, pose_link="floor_gripper")
        
        while not self._plan_and_execute(self._ariac_robots, self._floor_robot, self.get_logger(), "floor_robot"):
            pass
    
    def _move_floor_robot_2_to_pose(self,pose : Pose):
        with self._planning_scene_monitor.read_write() as scene:
            self._floor_robot_2.set_start_state(robot_state = scene.current_state)

            pose_goal = PoseStamped()
            pose_goal.header.frame_id = "world"
            pose_goal.pose = pose
            self._floor_robot_2.set_goal_state(pose_stamped_msg=pose_goal, pose_link="floor_2_gripper")
        
        while not self._plan_and_execute(self._ariac_robots, self._floor_robot_2, self.get_logger(), "floor_robot_2"):
            pass
    
    def bin_1_camera_cb(self,msg : AdvancedLogicalCameraImageMsg):
        self.bin_1_parts = msg.part_poses
        self.bin_1_camera_pose = msg.sensor_pose
    
    def bin_2_camera_cb(self,msg : AdvancedLogicalCameraImageMsg):
        self.bin_2_parts = msg.part_poses
        self.bin_2_camera_pose = msg.sensor_pose
    
    def _floor_robot_wait_for_attach(self,timeout : float, orientation : Quaternion):
        with self._planning_scene_monitor.read_write() as scene:
            current_pose = scene.current_state.get_pose("floor_gripper")
        self.get_logger().info("Got current pose")
        start_time = time.time()
        while not self._floor_robot_gripper_state.attached:
            sleep(0.2)
            if time.time()-start_time>=timeout:
                self.get_logger().error("Unable to pick up part")
                return False
            current_pose=build_pose(current_pose.position.x, current_pose.position.y,
                                    current_pose.position.z-0.0005,
                                    orientation)
            waypoints = [current_pose]
            self._move_floor_robot_cartesian(waypoints, 0.3, 0.3, False)
            
        self.get_logger().info("Attached to part")
        return True

    def _floor_robot_2_wait_for_attach(self,timeout : float, orientation : Quaternion):
        with self._planning_scene_monitor.read_write() as scene:
            current_pose = scene.current_state.get_pose("floor_2_gripper")
        self.get_logger().info("Got current pose")
        start_time = time.time()
        while not self._floor_robot_2_gripper_state.attached:
            sleep(0.2)
            if time.time()-start_time>=timeout:
                self.get_logger().error("Unable to pick up part")
                return False
            current_pose=build_pose(current_pose.position.x, current_pose.position.y,
                                    current_pose.position.z-0.0005,
                                    orientation)
            waypoints = [current_pose]
            self._move_floor_robot_2_cartesian(waypoints, 0.3, 0.3, False)
            
        self.get_logger().info("Attached to part")
        return True

    def floor_robot_pick_bin_part(self,part_to_pick : PartMsg, robot = "floor_robot"):
        part_pose = Pose()
        found_part = False
        bin_side = ""
        for part in self.bin_1_parts:
            part : PartPoseMsg
            if (part.part.type == part_to_pick.type and part.part.color == part_to_pick.color) and robot == "floor_robot":
                part_pose = multiply_pose(self.bin_1_camera_pose,part.pose)
                found_part = True
                bin_side = "bin_1"
                break
        
        if not found_part:
            for part in self.bin_2_parts:
                part : PartPoseMsg
                if (part.part.type == part_to_pick.type and part.part.color == part_to_pick.color) and robot == "floor_robot_2": 
                    part_pose = multiply_pose(self.bin_2_camera_pose,part.pose)
                    found_part = True
                    bin_side = "bin_2"
                    break
        
        if not found_part:
            self.get_logger().error("Unable to locate part")
            return False
        else:
            self.get_logger().info(f"Part found in {bin_side}")

        part_rotation = rpy_from_quaternion(part_pose.orientation)[2]
        
        if robot == "floor_robot":
            gripper_orientation = quaternion_from_euler(0.0,pi,part_rotation)
            self._move_floor_robot_to_pose(build_pose(part_pose.position.x, part_pose.position.y,
                                                    part_pose.position.z+0.5, gripper_orientation))

            waypoints = [build_pose(part_pose.position.x, part_pose.position.y,
                                    part_pose.position.z+CompetitionInterface._part_heights[part_to_pick.type]+0.008,
                                    gripper_orientation)]
            self._move_floor_robot_cartesian(waypoints, 0.3, 0.3, False)
            self.set_floor_robot_gripper_state(True)
            self._floor_robot_wait_for_attach(30.0, gripper_orientation)

            self.floor_robot_move_to_joint_position("floor_robot_home")

            self.floor_robot_attached_part_ = part_to_pick
        else:
            gripper_orientation = quaternion_from_euler(0.0,pi,part_rotation)
            self._move_floor_robot_2_to_pose(build_pose(part_pose.position.x, part_pose.position.y,
                                                    part_pose.position.z+0.5, gripper_orientation))

            waypoints = [build_pose(part_pose.position.x, part_pose.position.y,
                                    part_pose.position.z+CompetitionInterface._part_heights[part_to_pick.type]+0.008,
                                    gripper_orientation)]
            self._move_floor_robot_2_cartesian(waypoints, 0.3, 0.3, False)
            self.set_floor_robot_gripper_state(True)
            self._floor_robot_2_wait_for_attach(30.0, gripper_orientation)

            self.floor_robot_2_move_to_joint_position("floor_robot_2_home")

            self.floor_robot_2_attached_part_ = part_to_pick
        return True
    
    def complete_orders(self):
        while len(self._orders) == 0:
            self.get_logger().info("No orders have been recieved yet", throttle_duration_sec=5.0)

        self.add_objects_to_planning_scene()

        success = True
        while True:
            if (self._competition_state == CompetitionStateMsg.ENDED):
                success = False
                break

            if len(self._orders) == 0:
                if (self._competition_state != CompetitionStateMsg.ORDER_ANNOUNCEMENTS_DONE):
                    self.get_logger().info("Waiting for orders...")
                    while len(self._orders) == 0:
                        sleep(1)
                else:
                    self.get_logger().info("Completed all orders")
                    success = True
                    break

            self.current_order = copy(self._orders[0])
            self.current_order : Order
            del self._orders[0]

            if self.current_order.order_type == OrderMsg.KITTING:
                self.complete_kitting_order(self.current_order.order_task)
                # while agv_location !=AGVStatusMsg.WAREHOUSE:
                #     agv_location = self._agv_locations[kitting_agv_num]
            elif self.current_order.order_type == OrderMsg.ASSEMBLY:
                self.complete_assembly_order(self.current_order.order_task)
            else:
                self.complete_combined_order(self.current_order.order_task)
            
            self.submit_order(self.current_order.order_id)
        self.end_competition()
        return success

    def complete_kitting_order(self, kitting_task:KittingTask):
        self._floor_robot_pick_and_place_tray(kitting_task._tray_id, kitting_task._agv_number)

        for kitting_part in kitting_task._parts:
            found = self.floor_robot_pick_bin_part(kitting_part._part)
            if not found:
                self.floor_robot_pick_conveyor_part(kitting_part.part)
            self._floor_robot_place_part_on_kit_tray(kitting_task._agv_number, kitting_part.quadrant)
        
        self.move_agv(kitting_task._agv_number, kitting_task._destination)

    def _floor_robot_pick_and_place_tray(self, tray_id, agv_number):
        tray_pose = Pose
        station = ""
        found_tray = False

        for tray in self._kts1_trays:
            if tray.id == tray_id:
                station = "kts1"
                tray_pose = multiply_pose(self._kts1_camera_pose, tray.pose)
                found_tray = True
                break
        
        if not found_tray:
            for tray in self._kts2_trays:
                if tray.id == tray_id:
                    station = "kts2"
                    tray_pose = multiply_pose(self._kts2_camera_pose, tray.pose)
                    found_tray = True
                    break
        
        if not found_tray:
            return False
        
        tray_rotation = rpy_from_quaternion(tray_pose.orientation)[2]

        self.floor_robot_move_to_joint_position(f"floor_{station}_js_")
        
        gripper_orientation = quaternion_from_euler(0.0,pi,tray_rotation)
        
        waypoints = [build_pose(tray_pose.position.x, tray_pose.position.y,
                                tray_pose.position.z+0.5, 
                                gripper_orientation),
                    build_pose(tray_pose.position.x, tray_pose.position.y,
                                tray_pose.position.z+0.003,
                                gripper_orientation)]
        self._move_floor_robot_cartesian(waypoints, 0.3, 0.3, False)
        self.set_floor_robot_gripper_state(True)
        self._floor_robot_wait_for_attach(5.0, gripper_orientation)
                
        waypoints = [build_pose(tray_pose.position.x, tray_pose.position.y,
                                tray_pose.position.z+0.2,
                                gripper_orientation)]
        self._move_floor_robot_cartesian(waypoints, 0.05, 0.05, False)

        self.floor_robot_move_to_joint_position(f"agv{agv_number}")

        agv_tray_pose = self._frame_world_pose(f"agv{agv_number}_tray")
        agv_rotation = rpy_from_quaternion(agv_tray_pose.orientation)[2]

        agv_rotation = quaternion_from_euler(0.0,pi,agv_rotation)

        self._move_floor_robot_to_pose(build_pose(agv_tray_pose.position.x, agv_tray_pose.position.y,
                                                  agv_tray_pose.position.z+0.5,agv_rotation))
        
        waypoints = [build_pose(agv_tray_pose.position.x, agv_tray_pose.position.y,
                                agv_tray_pose.position.z+0.01,agv_rotation)]
        
        self._move_floor_robot_cartesian(waypoints, 0.3, 0.3, False)
        self.set_floor_robot_gripper_state(False)
        self.lock_agv_tray(agv_number)

        waypoints = [build_pose(agv_tray_pose.position.x, agv_tray_pose.position.y,
                                agv_tray_pose.position.z+0.3,quaternion_from_euler(0.0,pi,0.0))]
        self._move_floor_robot_cartesian(waypoints,0.3,0.3)

    
    def _frame_world_pose(self,frame_id : str):
        self.get_logger().info(f"Getting transform for frame: {frame_id}")
        # try:
        t = self.tf_buffer.lookup_transform("world",frame_id,rclpy.time.Time())
        # except:
        #     self.get_logger().error("Could not get transform")
        #     quit()
        
        pose = Pose()
        pose.position.x = t.transform.translation.x
        pose.position.y = t.transform.translation.y
        pose.position.z = t.transform.translation.z
        pose.orientation = t.transform.rotation

        return pose
        
    def _floor_robot_place_part_on_kit_tray(self, agv_num : int, quadrant : int):
        
        if not self._floor_robot_gripper_state.attached:
            self.get_logger().error("No part attached")
            return False

        # self.floor_robot_move_joints_dict({"linear_actuator_joint":self._rail_positions[f"agv{agv_num}"],
        #                                "floor_shoulder_pan_joint":0})
        
        self.floor_robot_move_to_joint_position(f"agv{agv_num}")
        
        agv_tray_pose = self._frame_world_pose(f"agv{agv_num}_tray")

        part_drop_offset = build_pose(CompetitionInterface._quad_offsets[quadrant][0],
                                      CompetitionInterface._quad_offsets[quadrant][1],
                                      0.0, quaternion_from_euler(0.0,pi,0.0))
        
        part_drop_pose = multiply_pose(agv_tray_pose, part_drop_offset)

        self._move_floor_robot_to_pose(build_pose(part_drop_pose.position.x, part_drop_pose.position.y,
                                                  part_drop_pose.position.z+0.3, quaternion_from_euler(0.0, pi, 0.0)))
        
        waypoints = [build_pose(part_drop_pose.position.x, part_drop_pose.position.y,
                                part_drop_pose.position.z+CompetitionInterface._part_heights[self.floor_robot_attached_part_.type]+0.025, 
                                quaternion_from_euler(0.0, pi, 0.0))]
        
        self._move_floor_robot_cartesian(waypoints, 0.3, 0.3,False)

        self.set_floor_robot_gripper_state(False)

        self._remove_model_from_floor_gripper()

        waypoints = [build_pose(part_drop_pose.position.x, part_drop_pose.position.y,
                                part_drop_pose.position.z+0.3, 
                                quaternion_from_euler(0.0, pi, 0.0))]
        
        self._move_floor_robot_cartesian(waypoints, 0.3, 0.3,False)

        return True
                    
    def _makeAttachedMesh(self, name, pose, filename, robot) -> AttachedCollisionObject:
        with pyassimp.load(filename) as scene:
            assert len(scene.meshes)
            
            mesh = Mesh()
            for face in scene.meshes[0].faces:
                triangle = MeshTriangle()
                if hasattr(face, 'indices'):
                    if len(face.indices) == 3:
                        triangle.vertex_indices = [face.indices[0],
                                                    face.indices[1],
                                                    face.indices[2]]
                else:
                    if len(face) == 3:
                        triangle.vertex_indices = [face[0],
                                                    face[1],
                                                    face[2]]
                mesh.triangles.append(triangle)
            for vertex in scene.meshes[0].vertices:
                point = Point()
                point.x = float(vertex[0])
                point.y = float(vertex[1])
                point.z = float(vertex[2])
                mesh.vertices.append(point)
            
        o = AttachedCollisionObject()
        if robot == "floor_robot":
            o.link_name = "floor_gripper"
        else:
            o.link_name = "floor_2_gripper"
        o.object.header.frame_id = "world"
        o.object.id = name
        o.object.meshes.append(mesh)
        o.object.mesh_poses.append(pose)
        return o
    
    def apply_planning_scene(self, scene):
        apply_planning_scene_client = self.create_client(ApplyPlanningScene, "/apply_planning_scene")

        # Create a request object.
        request = ApplyPlanningScene.Request()

        # Set the request location.
        request.scene = scene

        # Send the request.
        future = apply_planning_scene_client.call_async(request)

        # Wait for the server to respond.
        while not future.done():
            pass

        # Check the result of the service call.
        if future.result().success:
            self.get_logger().info(f'Succssefully applied new planning scene')
        else:
            self.get_logger().warn(future.result().message)
    
    def get_planning_scene_msg(self, msg:PlanningScene) -> PlanningScene:
        self.planning_scene_msg = msg
    
    def floor_robot_move_to_joint_position(self, position_name : str):
        with self._planning_scene_monitor.read_write() as scene:
            self._floor_robot.set_start_state(robot_state=scene.current_state)
            scene.current_state.joint_positions = self.floor_position_dict[position_name]
            joint_constraint = construct_joint_constraint(
                    robot_state=scene.current_state,
                    joint_model_group=self._ariac_robots.get_robot_model().get_joint_model_group("floor_robot"),
            )
            self._floor_robot.set_goal_state(motion_plan_constraints=[joint_constraint])
        self._plan_and_execute(self._ariac_robots,self._floor_robot, self.get_logger(), robot_type="floor_robot")

    def floor_robot_2_move_to_joint_position(self, position_name : str):
        with self._planning_scene_monitor.read_write() as scene:
            self._floor_robot_2.set_start_state(robot_state=scene.current_state)
            scene.current_state.joint_positions = self.floor_position_dict[position_name]
            joint_constraint = construct_joint_constraint(
                    robot_state=scene.current_state,
                    joint_model_group=self._ariac_robots.get_robot_model().get_joint_model_group("floor_robot_2"),
            )
            self._floor_robot_2.set_goal_state(motion_plan_constraints=[joint_constraint])
        self._plan_and_execute(self._ariac_robots,self._floor_robot_2, self.get_logger(), robot_type="floor_robot_2")
  
    def _create_floor_joint_position_state(self, joint_positions : list)-> dict:
        return {"linear_actuator_joint":joint_positions[0],
                "floor_shoulder_pan_joint":joint_positions[1],
                "floor_shoulder_lift_joint":joint_positions[2],
                "floor_elbow_joint":joint_positions[3],
                "floor_wrist_1_joint":joint_positions[4],
                "floor_wrist_2_joint":joint_positions[5],
                "floor_wrist_3_joint":joint_positions[6]}

    def _create_floor_joint_position_dict(self, dict_positions = {}):
        with self._planning_scene_monitor.read_write() as scene:
            current_positions = scene.current_state.get_joint_group_positions("floor_robot")
            current_position_dict = self._create_floor_joint_position_state(current_positions)
            for key in dict_positions.keys():
                current_position_dict[key] = dict_positions[key]
        return current_position_dict

    def floor_robot_move_joints_dict(self, dict_positions : dict):
        new_joint_position = self._create_floor_joint_position_dict(dict_positions)
        with self._planning_scene_monitor.read_write() as scene:
            self._floor_robot.set_start_state(robot_state = scene.current_state)
            scene.current_state.joint_positions = new_joint_position
            joint_constraint = construct_joint_constraint(
                    robot_state=scene.current_state,
                    joint_model_group=self._ariac_robots.get_robot_model().get_joint_model_group("floor_robot"),
            )
            self._floor_robot.set_goal_state(motion_plan_constraints=[joint_constraint])
        self._plan_and_execute(self._ariac_robots,self._floor_robot, self.get_logger(), "floor_robot")

    def fromMsg(self, pose):
        o1 = pose.orientation
        p1 = pose.position
        return PyKDL.Frame(PyKDL.Rotation.Quaternion(o1.x, o1.y, o1.z, o1.w),
                           PyKDL.Vector(p1.x, p1.y, p1.z))

    def toMsg(self, frame):
        pose = Pose()
        pose.position.x = frame.p.x()
        pose.position.y = frame.p.y()
        pose.position.z = frame.p.z()

        q = frame.M.GetQuaternion()
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]

        return pose

    def conveyor_parts_cb(self, msg):
        self.conveyor_parts_expected = msg.parts
    
    def conveyor_camera_cb(self, msg : AdvancedLogicalCameraImageMsg):
        self.last_conveyor_image_time = self.get_clock().now().nanoseconds
        self.conveyor_camera_image = msg
    
    def breakbeam_cb(self, msg : BreakBeamStatus):
        if not self.breakbeam_recieved_data:
            self.breakbeam_recieved_data = True
            self.breakbeam_pose = self._frame_world_pose(msg.header.frame_id)
        
        self.info_log(f"Conveyor images part poses length: {len(self.conveyor_camera_image.part_poses)}")
        if self.conveyor_camera_image is not None and msg.object_detected and len(self.conveyor_camera_image.part_poses)>0:
            part_y_positions = [p.pose.position.y for p in self.conveyor_camera_image.part_poses]
            part_pose = self.conveyor_camera_image.part_poses[part_y_positions.index(max(part_y_positions))]
            part_world_pose = multiply_pose(self.conveyor_camera_image.sensor_pose, part_pose.pose)
            self.conveyor_parts.append(ConveyorPart(part_pose.part, part_world_pose, self.last_conveyor_image_time))
                
        
    def get_time_at_pick_position(self, conv_part : ConveyorPart):
        time_to_y_zero = abs(conv_part.pose.position.y)/self.conveyor_speed
        return conv_part.detection_time + (time_to_y_zero * 10**9)
    
    def floor_robot_pick_conveyor_part(self, part_to_pick : PartMsg):
        if len(self.conveyor_parts_expected)==0:
            self.get_logger().info("No parts expected on the conveyor belt")
            return
        for part in self.conveyor_parts_expected:
            if part.part.type == part_to_pick.type and part.part.color == part_to_pick.color:
                self.get_logger().info(f"Attepting to pick a {self._part_colors[part_to_pick.color]} {self._part_types[part_to_pick.type]} from the conveyor")
                break
            elif part == self.conveyor_parts_expected[-1]:
                self.get_logger().info("Unable to locate part on the conveyor")
                return
        
        found_part = False
        part_picked = False
        num_tries = 0

        if self._floor_robot_gripper_state.type != "part_gripper":
            with self._planning_scene_monitor.read_write() as scene:
                current_pose = scene.current_state.get_pose("floor_gripper")
            s = "1"
            if current_pose.position.y > 0:
                s = "2"
            self.floor_robot_move_to_joint_position(f"floor_kts{s}_js_")
            self._floor_robot_change_gripper(f"kts{s}", "parts")
        
        self.get_logger().info("Moving floor robot to conveyor pick location")

        while num_tries < 3:
            self.floor_robot_move_to_joint_position("floor_conveyor_js_")
        
            # Wait until a part of the correct type and color is on the conveyor
            self.info_log(f"Waiting for {self._part_colors[part_to_pick.color]} {self._part_types[part_to_pick.type]} to arrive on the conveyor")
            while not found_part:
                if self.conveyor_parts:
                    for conveyor_part in self.conveyor_parts:
                        conveyor_part : ConveyorPart
                        if conveyor_part.part.type == part_to_pick.type and conveyor_part.part.color == part_to_pick.color:
                            pick_time_for_part = self.get_time_at_pick_position(conveyor_part)
                            self.info_log(f"Time at pick position: {pick_time_for_part/10**9}")
                            if pick_time_for_part-self.get_clock().now().nanoseconds > 5 * (10**9):
                                self.info_log("Found part on conveyor")
                                part_pose = conveyor_part.pose
                                # self.info_log(f"Time at pick position: {pick_time_for_part/10**9}")
                                found_part = True
                                break
            
            # Move to pre-pick position
            offset = 0.15
            part_rotation = rpy_from_quaternion(part_pose.orientation)[2]
            
            waypoints = [build_pose(part_pose.position.x, 0.0, part_pose.position.z + offset, 
                                    quaternion_from_euler(0.0, pi, part_rotation))]
            
            self._move_floor_robot_cartesian(waypoints, 0.5, 0.5, False)
            
            self.info_log("Ready to pick part from conveyor")

            # Plan trajectory to pick part
            waypoints = [build_pose(part_pose.position.x, 0.0,
                                    part_pose.position.z + self._part_heights[part_to_pick.type] - (0.002 if part_to_pick.type != PartMsg.PUMP else -0.002), 
                                    quaternion_from_euler(0.0, pi, part_rotation))]
                
            trajectory_msg = self._call_get_cartesian_path(waypoints, 1.0, 1.0, False, "floor_robot")
            with self._planning_scene_monitor.read_write() as scene:
                trajectory = RobotTrajectory(self._ariac_robots.get_robot_model())
                trajectory.set_robot_trajectory_msg(scene.current_state, trajectory_msg)
                trajectory.joint_model_group_name = "floor_robot"
                scene.current_state.update(True)
                self._ariac_robots_state = scene.current_state
            
            # Calculate the duration of the movement
            trajectory_time = Duration.from_msg(trajectory.get_robot_trajectory_msg().joint_trajectory.points[-1].time_from_start)
            buffer = Duration(seconds=0.3) 

            # Wait for the part to arrive at pick location
            self.set_floor_robot_gripper_state(True)
            self.info_log("Waiting for part to arrive")
            while self.get_clock().now().nanoseconds < (pick_time_for_part - trajectory_time.nanoseconds - buffer.nanoseconds):
                pass

            # Execute picking trajectory
            self.info_log("Executing pick movement")
            self._ariac_robots.execute(trajectory, controllers=[])

            # Move up
            self.info_log("Moving up")
            waypoints = [build_pose(part_pose.position.x, 0.0, part_pose.position.z + 0.5, 
                                    quaternion_from_euler(0.0, pi, part_rotation))]
            self._move_floor_robot_cartesian(waypoints, 1.0, 1.0, False)

            # Check that part is attached
            timeout = 3.0
            start_time = time.time()
            attached = False
            self.info_log("Before while loop")
            while timeout>time.time()-start_time:
                if (self._floor_robot_gripper_state.attached):
                    self.info_log("Part picked successfully")
                    self.floor_robot_attached_part_ = part_to_pick
                    attached = True
                    break
            self.info_log("After while loop")
            if attached:
                break
            num_tries += 1
            
        return True

    def info_log(self, msg):
        self.get_logger().info(str(msg))
    
    def async_plan_execute(self,
                      robot,
                      planning_component,
                      logger,
                      robot_type
                      ):
        plan_and_execute_partial = partial(self._plan_and_execute, robot, planning_component, logger, robot_type)
        plan_and_execute_thread = threading.Thread(target=plan_and_execute_partial)
        plan_and_execute_thread.run()
    
    def async_execute(self,trajectory, controllers):
        execute_partial = partial(self.ex, trajectory, controllers)
        execute_thread = threading.Thread(target = execute_partial)
        execute_thread.run()

    def ex(self, trajectory, controllers):
        self._ariac_robots.execute(trajectory, controllers=controllers)