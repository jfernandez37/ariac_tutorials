from argparse import _MutuallyExclusiveGroup
from time import sleep
from math import pi
from copy import copy
import time
from sympy import Quaternion
from ament_index_python import get_package_share_directory
from moveit import MoveItPy, PlanningSceneMonitor
import rclpy
import pyassimp
import yaml
from rclpy.time import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.parameter import Parameter
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup

from geometry_msgs.msg import PoseStamped, Pose, Point, TransformStamped
from shape_msgs.msg import Mesh, MeshTriangle
from moveit_msgs.msg import CollisionObject
from std_msgs.msg import Header

from moveit.core.robot_trajectory import RobotTrajectory
from moveit.core.robot_state import RobotState, robotStateToRobotStateMsg
from moveit_msgs.srv import GetCartesianPath, GetPositionFK

from ariac_msgs.msg import (
    CompetitionState as CompetitionStateMsg,
    BreakBeamStatus as BreakBeamStatusMsg,
    AdvancedLogicalCameraImage as AdvancedLogicalCameraImageMsg,
    Part as PartMsg,
    PartPose as PartPoseMsg,
    Order as OrderMsg,
    AssemblyPart as AssemblyPartMsg,
    AGVStatus as AGVStatusMsg,
    AssemblyTask as AssemblyTaskMsg,
    VacuumGripperState,
)

from ariac_msgs.srv import (
    MoveAGV,
    VacuumGripperControl,
    ChangeGripper,
    SubmitOrder
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

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster


class Error(Exception):
  def __init__(self, value: str):
      self.value = value

  def __str__(self):
      return repr(self.value)
  
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

    def __init__(self):
        super().__init__('competition_interface')

        sim_time = Parameter(
            "use_sim_time",
            rclpy.Parameter.Type.BOOL,
            True
        )

        self.set_parameters([sim_time])
        
        # ROS2 callback groups
        self.ariac_cb_group = MutuallyExclusiveCallbackGroup()
        self.moveit_cb_group = MutuallyExclusiveCallbackGroup()
        self.orders_cb_group = ReentrantCallbackGroup()

        # Service client for starting the competition
        self._start_competition_client = self.create_client(Trigger, '/ariac/start_competition')

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
        self._parse_incoming_order = False
        
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

        # Attribute to store the current state of the floor robot gripper
        self._floor_robot_gripper_state = VacuumGripperState()

        # Moveit_py variables
        self._ariac_robots = MoveItPy(node_name="ariac_robots_moveit_py")
        self._ariac_robots_state = RobotState(self._ariac_robots.get_robot_model())

        self._floor_robot = self._ariac_robots.get_planning_component("floor_robot")
        self._ceiling_robot = self._ariac_robots.get_planning_component("ceiling_robot")

        self._floor_robot_home_quaternion = Quaternion()
        self._ceiling_robot_home_quaternion = Quaternion()

        self._planning_scene_monitor = self._ariac_robots.get_planning_scene_monitor()

        # Parts found in the bins
        self._left_bins_parts = []
        self._right_bins_parts = []
        self._left_bins_camera_pose = Pose()
        self._right_bins_camera_pose = Pose()

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
                                                             "/ariac/sensors/left_bins_camera/image",
                                                             self._left_bins_camera_cb,
                                                             qos_profile_sensor_data,
                                                             callback_group=self.moveit_cb_group)
        self.right_bins_camera_sub = self.create_subscription(AdvancedLogicalCameraImageMsg,
                                                             "/ariac/sensors/right_bins_camera/image",
                                                             self._right_bins_camera_cb,
                                                             qos_profile_sensor_data,
                                                             callback_group=self.moveit_cb_group)
        self.kts1_camera_sub_ = self.create_subscription(AdvancedLogicalCameraImageMsg,
                                                         "/ariac/sensors/kts1_camera/image",
                                                         self._kts1_camera_cb,
                                                         qos_profile_sensor_data,
                                                             callback_group=self.moveit_cb_group)
        self.kts2_camera_sub_ = self.create_subscription(AdvancedLogicalCameraImageMsg,
                                                         "/ariac/sensors/kts2_camera/image",
                                                         self._kts2_camera_cb,
                                                         qos_profile_sensor_data,
                                                         callback_group=self.moveit_cb_group)
        
        # AGV status subs
        self._agv_locations = {1 : -1,
                               2 : -1,
                               3 : -1,
                               4 : -1}
        
        self.agv1_status_sub = self.create_subscription(AGVStatusMsg,
                                                        "/ariac/agv1_status",
                                                        self._agv1_status_cb,
                                                        10,
                                                        callback_group=self.moveit_cb_group)
        self.agv2_status_sub = self.create_subscription(AGVStatusMsg,
                                                        "/ariac/agv2_status",
                                                        self._agv2_status_cb,
                                                        10,
                                                        callback_group=self.moveit_cb_group)
        self.agv3_status_sub = self.create_subscription(AGVStatusMsg,
                                                        "/ariac/agv3_status",
                                                        self._agv3_status_cb,
                                                        10,
                                                        callback_group=self.moveit_cb_group)
        self.agv4_status_sub = self.create_subscription(AGVStatusMsg,
                                                        "/ariac/agv4_status",
                                                        self._agv4_status_cb,
                                                        10,
                                                        callback_group=self.moveit_cb_group)
        
        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.tf_broadcaster = StaticTransformBroadcaster(self)
        self.static_transforms = []

        self.floor_robot_attached_part_ = PartMsg()

        self._change_gripper_client = self.create_client(ChangeGripper, "/ariac/floor_robot_change_gripper")


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

    def _advanced_camera0_cb(self, msg: AdvancedLogicalCameraImageMsg):
        '''Callback for the topic /ariac/sensors/advanced_camera_0/image

        Arguments:
            msg -- AdvancedLogicalCameraImage message
        '''
        self._camera_image = AdvancedLogicalCameraImage(msg.part_poses,
                                                        msg.tray_poses,
                                                        msg.sensor_pose)

    def _breakbeam0_cb(self, msg: BreakBeamStatusMsg):
        '''Callback for the topic /ariac/sensors/breakbeam_0/status

        Arguments:
            msg -- BreakBeamStatusMsg message
        '''
        if not self._object_detected and msg.object_detected:
            self._conveyor_part_count += 1

        self._object_detected = msg.object_detected

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
            # try:
            #     rclpy.spin_once(self)
            # except KeyboardInterrupt:
            #     return
            pass

        self.get_logger().info('Competition is ready. Starting...')

        # Check if service is available
        if not self._start_competition_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error('Service \'/ariac/start_competition\' is not available.')
            return

        # Create trigger request and call starter service
        request = Trigger.Request()
        future = self._start_competition_client.call_async(request)

        while not future.done():
            pass
        # Wait until the service call is completed
        # rclpy.spin_until_future_complete(self, future)

        if future.result().success:
            self.get_logger().info('Started competition.')
        else:
            self.get_logger().warn('Unable to start competition')

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

    def lock_agv_tray(self, num):
        '''
        Lock the tray of an AGV and parts on the tray. This will prevent tray and parts from moving during transport.
        Args:
            num (int):  AGV number
        Raises:
            KeyboardInterrupt: Exception raised when the user presses Ctrl+C
        '''

        # Create a client to send a request to the `/ariac/agv{num}_lock_tray` service
        tray_locker = self.create_client(
            Trigger,
            f'/ariac/agv{num}_lock_tray'
        )

        # Build the request
        request = Trigger.Request()
        # Send the request
        future = tray_locker.call_async(request)

        # Wait for the response
        try:
            rclpy.spin_until_future_complete(self, future)
        except KeyboardInterrupt as kb_error:
            raise KeyboardInterrupt from kb_error

        # Check the response
        if future.result().success:
            self.get_logger().info(f'Locked AGV{num}\'s tray')
        else:
            self.get_logger().warn('Unable to lock tray')

    def move_agv_to_station(self, num, station):
        '''
        Move an AGV to an assembly station.
        Args:
            num (int): AGV number
            station (int): Assembly station number
        Raises:
            KeyboardInterrupt: Exception raised when the user presses Ctrl+C
        '''

        # Create a client to send a request to the `/ariac/move_agv` service.
        mover = self.create_client(
            MoveAGV,
            f'/ariac/move_agv{num}')

        # Create a request object.
        request = MoveAGV.Request()

        # Set the request location.
        if station in [AssemblyTaskMsg.AS1, AssemblyTaskMsg.AS3]:
            request.location = MoveAGV.Request.ASSEMBLY_FRONT
        else:
            request.location = MoveAGV.Request.ASSEMBLY_BACK

        # Send the request.
        future = mover.call_async(request)

        # Wait for the server to respond.
        try:
            rclpy.spin_until_future_complete(self, future)
        except KeyboardInterrupt as kb_error:
            raise KeyboardInterrupt from kb_error

        # Check the result of the service call.
        if future.result().success:
            self.get_logger().info(f'Moved AGV{num} to {self._stations[station]}')
        else:
            self.get_logger().warn(future.result().message)  

    def set_floor_robot_gripper_state(self, state):
        '''Set the gripper state of the floor robot.

        Arguments:
            state -- True to enable, False to disable

        Raises:
            KeyboardInterrupt: Exception raised when the user presses Ctrl+C
        '''
        if self._floor_robot_gripper_state.enabled == state:
            self.get_logger().warn(f'Gripper is already {self._gripper_states[state]}')
            return

        request = VacuumGripperControl.Request()
        request.enable = state

        future = self._floor_gripper_enable.call_async(request)

        try:
            rclpy.spin_until_future_complete(self, future)
        except KeyboardInterrupt as kb_error:
            raise KeyboardInterrupt from kb_error

        if future.result().success:
            self.get_logger().info(f'Changed gripper state to {self._gripper_states[state]}')
        else:
            self.get_logger().warn('Unable to change gripper state')

    def wait(self, duration):
        '''Wait for a specified duration.

        Arguments:
            duration -- Duration to wait in seconds

        Raises:
            KeyboardInterrupt: Exception raised when the user presses Ctrl+C
        '''
        start = self.get_clock().now()

        while self.get_clock().now() <= start + Duration(seconds=duration):
            try:
                rclpy.spin_once(self)
            except KeyboardInterrupt as kb_error:
                raise KeyboardInterrupt from kb_error
    
    def _call_get_cartesian_path (self, waypoints : list, 
                                  max_velocity_scaling_factor : float, 
                                  max_acceleration_scaling_factor : float):

        self.get_logger().info("Getting cartesian path")
        self._ariac_robots_state.update()

        request = GetCartesianPath.Request()

        header = Header()
        header.frame_id = "world"
        header.stamp = self.get_clock().now().to_msg()

        request.header = header
        request.start_state = robotStateToRobotStateMsg(self._ariac_robots_state)
        request.group_name = "floor_robot"
        request.link_name = "floor_gripper"
        request.waypoints = waypoints
        request.max_step = 0.2
        request.avoid_collisions = True
        request.max_velocity_scaling_factor = max_velocity_scaling_factor
        request.max_acceleration_scaling_factor = max_acceleration_scaling_factor

        
        future = self.get_cartesian_path_client.call_async(request)

        rclpy.spin_until_future_complete(self, future, timeout_sec=10)


        if not future.done():
            raise Error("Timeout reached when calling move_cartesian service")

        result: GetCartesianPath.Response
        result = future.result()

        return result.solution

    def _call_get_position_fk (self):

        request = GetPositionFK.Request()


        header = Header()
        header.frame_id = "world"
        header.stamp = self.get_clock().now().to_msg()
        request.header = header


        request.fk_link_names = ["floor_gripper"]
        request.robot_state = robotStateToRobotStateMsg(self._ariac_robots_state)

        future = self.get_position_fk_client.call_async(request)


        rclpy.spin_until_future_complete(self, future, timeout_sec=10)

        if not future.done():
            raise Error("Timeout reached when calling get_position_fk service")

        result: GetPositionFK.Response
        result = future.result()

        return result.pose_stamped[0].pose
    
    def _plan_and_execute(
        self,
        robot,
        planning_component,
        logger,
        single_plan_parameters=None,
        multi_plan_parameters=None,
        sleep_time=0.0,
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
            robot_trajectory = plan_result.trajectory
            logger.info(str(robot_trajectory.joint_model_group_name))
            robot.execute(robot_trajectory, controllers=[])
        else:
            logger.error("Planning failed")

        sleep(sleep_time)

    def move_floor_robot_home(self):
        self._floor_robot.set_start_state_to_current_state()
        self._floor_robot.set_goal_state(configuration_name="home")
        self._plan_and_execute(self._ariac_robots,self._floor_robot, self.get_logger(), sleep_time=0.0)
        self._ariac_robots_state.update()
        self._floor_robot_home_quaternion = self._ariac_robots_state.get_pose("floor_gripper").orientation
    
    def move_ceiling_robot_home(self):
        self._ceiling_robot.set_start_state_to_current_state()
        self._ceiling_robot.set_goal_state(configuration_name="home")
        self._plan_and_execute(self._ariac_robots,self._ceiling_robot, self.get_logger(), sleep_time=0.0)
        self._ariac_robots_state.update()
        self._ceiling_robot_home_quaternion = self._ariac_robots_state.get_pose("ceiling_gripper").orientation

    def _move_floor_robot_cartesian(self, waypoints, velocity, acceleration):
        with self._planning_scene_monitor.read_write() as scene:
            # instantiate a RobotState instance using the current robot model
            self._ariac_robots_state = scene.current_state
            self._ariac_robots_state.update()

            # Max step
            self._ariac_robots_state.update()
            trajectory_msg = self._call_get_cartesian_path(waypoints, velocity, acceleration)
            self._ariac_robots_state.update()
            trajectory = RobotTrajectory(self._ariac_robots.get_robot_model())
            trajectory.set_robot_trajectory_msg(self._ariac_robots_state, trajectory_msg)
            trajectory.joint_model_group_name = "floor_robot"
        self._ariac_robots_state.update(True)
        self._ariac_robots.execute(trajectory, controllers=[])

    def _move_floor_robot_to_pose(self,pose : Pose):
        self.get_logger().info(str(pose))
        with self._planning_scene_monitor.read_write() as scene:
            self._floor_robot.set_start_state_to_current_state()

            pose_goal = PoseStamped()
            pose_goal.header.frame_id = "world"
            pose_goal.pose = pose
            self.get_logger().info(str(pose_goal.pose))
            self._floor_robot.set_goal_state(pose_stamped_msg=pose_goal, pose_link="floor_gripper")
        
        self._plan_and_execute(self._ariac_robots, self._floor_robot, self.get_logger())

    def _makeMesh(self, name, pose, filename) -> CollisionObject:
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
            
        o = CollisionObject()
        o.header.frame_id = "world"
        o.id = name
        o.meshes.append(mesh)
        o.mesh_poses.append(pose)
        o.operation = o.ADD
        return o
    
    def _add_model_to_planning_scene(self,
                                    name : str,
                                    mesh_file : str,
                                    model_pose : Pose
                                    ):
        self.get_logger().info(f"Adding {name} to planning scene")
        package_share_directory = get_package_share_directory("test_competitor")
        model_path = package_share_directory + "/meshes/"+mesh_file
        collision_object = self._makeMesh(name, model_pose,model_path)
        with self._planning_scene_monitor.read_write() as scene:
            scene.apply_collision_object(collision_object)
            scene.current_state.update()
    
    def add_objects_to_planning_scene(self):
        package_share_directory = get_package_share_directory("ariac_tutorials")
        with open(package_share_directory+"/config/collision_object_info.yaml",'r') as object_file:
            objects_dict = yaml.safe_load(object_file)
        
        objects_dict : dict
        for key in objects_dict.keys():

            object_pose = Pose()
            
            object_pose.position.x = float(objects_dict[key]["position"][0])
            object_pose.position.y = float(objects_dict[key]["position"][1])
            object_pose.position.z = float(objects_dict[key]["position"][2])
            
            object_pose.orientation.x = float(objects_dict[key]["orientation"][0])
            object_pose.orientation.y = float(objects_dict[key]["orientation"][1])
            object_pose.orientation.z = float(objects_dict[key]["orientation"][2])
            object_pose.orientation.w = float(objects_dict[key]["orientation"][3])

            self._add_model_to_planning_scene(key, objects_dict[key]["file"], object_pose)
    
    def _left_bins_camera_cb(self,msg : AdvancedLogicalCameraImageMsg):
        self._left_bins_parts = msg.part_poses
        self._left_bins_camera_pose = msg.sensor_pose
    
    def _right_bins_camera_cb(self,msg : AdvancedLogicalCameraImageMsg):
        self._right_bins_parts = msg.part_poses
        self._right_bins_camera_pose = msg.sensor_pose
    
    def _kts1_camera_cb(self, msg: AdvancedLogicalCameraImageMsg):
        self._kts1_trays = msg.tray_poses
        self._kts1_camera_pose = msg.sensor_pose

    def _kts2_camera_cb(self, msg: AdvancedLogicalCameraImageMsg):
        self._kts2_trays = msg.tray_poses
        self._kts2_camera_pose = msg.sensor_pose
    
    def _agv1_status_cb(self, msg : AGVStatusMsg):
        self._agv_locations[1] = msg.location
    
    def _agv2_status_cb(self, msg : AGVStatusMsg):
        self._agv_locations[2] = msg.location
    
    def _agv3_status_cb(self, msg : AGVStatusMsg):
        self._agv_locations[3] = msg.location
    
    def _agv4_status_cb(self, msg : AGVStatusMsg):
        self._agv_locations[4] = msg.location

    def _floor_robot_wait_for_attach(self,timeout : float, orientation : Quaternion):
        with self._planning_scene_monitor.read_write() as scene:
            current_pose = scene.current_state.get_pose("floor_gripper")
        self.get_logger().info("Got current pose")
        start_time = time.time()
        while not self._floor_robot_gripper_state.attached:
            current_pose=build_pose(current_pose.position.x, current_pose.position.y,
                                    current_pose.position.z-0.001,
                                    orientation)
            waypoints = [current_pose]
            self._move_floor_robot_cartesian(waypoints, 0.3, 0.3)
            sleep(0.2)
            if time.time()-start_time>=timeout:
                self.get_logger().error("Unable to pick up part")

    def floor_robot_pick_bin_part(self,part_to_pick : PartMsg):
        part_pose = Pose()
        found_part = False
        bin_side = ""
        
        for part in self._left_bins_parts:
            part : PartPoseMsg
            if (part.part.type == part_to_pick.type and part.part.color == part_to_pick.color):
                part_pose = multiply_pose(self._left_bins_camera_pose,part.pose)
                found_part = True
                bin_side = "left_bins"
                break
        
        if not found_part:
            for part in self._right_bins_parts:
                part : PartPoseMsg
                if (part.part.type == part_to_pick.type and part.part.color == part_to_pick.color):
                    part_pose = multiply_pose(self._right_bins_camera_pose,part.pose)
                    found_part = True
                    bin_side = "right_bins"
                    break
        
        if not found_part:
            self.get_logger().error("Unable to locate part")
        else:
            self.get_logger().info(f"Part found in {bin_side}")

        part_rotation = rpy_from_quaternion(part_pose.orientation)[2]
        if self._floor_robot_gripper_state.type != "part_gripper":
            if part_pose.position.y<0:
                station = "kts1"
            else: 
                station = "kts2"
            self._floor_robot_change_gripper(station, "parts")
        gripper_orientation = quaternion_from_euler(0.0,pi,part_rotation)
        self._move_floor_robot_to_pose(build_pose(part_pose.position.x, part_pose.position.y,
                                                  part_pose.position.z+0.5, gripper_orientation))

        waypoints = [build_pose(part_pose.position.x, part_pose.position.y,
                                part_pose.position.z+CompetitionInterface._part_heights[part_to_pick.type]+0.008,
                                gripper_orientation)]
        self._move_floor_robot_cartesian(waypoints, 0.3, 0.3)
        self.set_floor_robot_gripper_state(True)
        self._floor_robot_wait_for_attach(30.0, gripper_orientation)
        self.floor_robot_attached_part_ = part_to_pick
        self.get_logger().info("Part attached. Attempting to move up")
        waypoints = [build_pose(part_pose.position.x, part_pose.position.y,
                                part_pose.position.z+0.5,
                                gripper_orientation)]
        self._move_floor_robot_cartesian(waypoints, 0.3, 0.3)
    
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
                if (self._competition_state == CompetitionStateMsg.ORDER_ANNOUNCEMENTS_DONE):
                    self.get_logger().info("Waiting for orders...")
                    while len(self._orders) == 0:
                        sleep(1)
                else:
                    self.get_logger().info("Completed all orders")
                    success = True
                    break

            current_order = copy(self._orders[0])
            current_order : Order
            del self._orders[0]
            kitting_agv_num = -1

            if current_order.order_type== OrderMsg.KITTING:
                self.complete_kitting_order(current_order.order_task)
                kitting_agv_num = current_order.order_task.agv_number
            else:
                self.get_logger().info(f"Unable to complete {'assembly' if current_order.order_type == OrderMsg.ASSEMBLY else 'combined'} order")
                return False
            agv_location = -1 
            
            while agv_location !=AGVStatusMsg.WAREHOUSE:
                agv_location = self._agv_locations[kitting_agv_num]
            
            self.submit_order(current_order.id)
        return success

    def complete_kitting_order(self, kitting_task:KittingTask):
        self.move_floor_robot_home()

        self._floor_robot_pick_and_place_tray(kitting_task._tray_id, kitting_task._agv_number)

        for kitting_part in kitting_task._parts:
            self.floor_robot_pick_bin_part(kitting_part._part)
            self._floor_robot_place_part_on_kit_tray(kitting_task._agv_number, kitting_part.quadrant)
        
        self.move_agv_to_station(kitting_task._agv_number, kitting_task._destination)

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

        if self._floor_robot_gripper_state.type != "tray_gripper":
            self._floor_robot_change_gripper(station, "trays")
        
        gripper_orientation = quaternion_from_euler(0.0,pi,tray_rotation)
        self._move_floor_robot_to_pose(build_pose(tray_pose.position.x, tray_pose.position.y,
                                                  tray_pose.position.z+0.5, gripper_orientation))
        
        waypoints = [build_pose(tray_pose.position.x, tray_pose.position.y,
                                tray_pose.position.z+0.001,
                                gripper_orientation)]
        self._move_floor_robot_cartesian(waypoints, 0.3, 0.3)
        self.set_floor_robot_gripper_state(True)
        self._floor_robot_wait_for_attach(30.0, gripper_orientation)
        waypoints = [build_pose(tray_pose.position.x, tray_pose.position.y,
                                tray_pose.position.z+0.5,
                                gripper_orientation)]
        self._move_floor_robot_cartesian(waypoints, 0.3, 0.3)

        agv_tray_pose = self._frame_world_pose(f"agv{agv_number}_tray")
        agv_rotation = rpy_from_quaternion(agv_tray_pose.orientation)[2]

        agv_quaternion = quaternion_from_euler(0.0,pi,agv_rotation)

        self._move_floor_robot_to_pose(build_pose(agv_tray_pose.position.x, agv_tray_pose.position.y,
                                                  agv_tray_pose.position.z+0.5,agv_quaternion))
        
        waypoints = [build_pose(agv_tray_pose.position.x, agv_tray_pose.position.y,
                                agv_tray_pose.position.z+0.01,agv_quaternion)]
        
        self._move_floor_robot_cartesian(waypoints, 0.3, 0.3)
        self.set_floor_robot_gripper_state(False)
        self.lock_agv_tray(agv_number)

        waypoints = [build_pose(agv_tray_pose.position.x, agv_tray_pose.position.y,
                                agv_tray_pose.position.z+0.3,quaternion_from_euler(0.0,pi,0.0))]

    
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
        
        agv_tray_pose = self._frame_world_pose(f"agv{agv_num}_tray")

        part_drop_offset = build_pose(CompetitionInterface._quad_offsets[quadrant][0],
                                      CompetitionInterface._quad_offsets[quadrant][1],
                                      0.0, quaternion_from_euler(0.0,pi,0.0))
        
        part_drop_pose = multiply_pose(agv_tray_pose, part_drop_offset)

        self._move_floor_robot_to_pose(build_pose(part_drop_pose.position.x, part_drop_pose.position.y,
                                                  part_drop_pose.position.z+0.3, quaternion_from_euler(0.0, pi, 0.0)))
        
        waypoints = [build_pose(part_drop_pose.position.x, part_drop_pose.position.y,
                                part_drop_pose.position.z+CompetitionInterface._part_heights[self.floor_robot_attached_part_.type]+0.01, 
                                quaternion_from_euler(0.0, pi, 0.0))]
        
        self._move_floor_robot_cartesian(waypoints, 0.3, 0.3)

        self.set_floor_robot_gripper_state(False)

        waypoints = [build_pose(part_drop_pose.position.x, part_drop_pose.position.y,
                                part_drop_pose.position.z+0.3, 
                                quaternion_from_euler(0.0, pi, 0.0))]
        
        self._move_floor_robot_cartesian(waypoints, 0.3, 0.3)

        return True

    def _floor_robot_change_gripper(self,station : str, gripper_type : str):
        tc_pose = self._frame_world_pose(f"{station}_tool_changer_{gripper_type}_frame")

        self._move_floor_robot_to_pose(build_pose(tc_pose.position.x, tc_pose.position.y,
                                                  tc_pose.position.z+0.4,
                                                  quaternion_from_euler(0.0, pi, 0.0)))
        
        waypoints = [build_pose(tc_pose.position.x, tc_pose.position.y,tc_pose.position.z,
                                quaternion_from_euler(0.0,pi,0.0))]
        
        self._move_floor_robot_cartesian(waypoints, 0.3, 0.3)

        request = ChangeGripper.Request()

        if gripper_type == "trays":
            request.gripper_type = ChangeGripper.Request.TRAY_GRIPPER
        elif gripper_type == "parts":
            request.gripper_type = ChangeGripper.Request.PART_GRIPPER
        
        future = self._change_gripper_client.call_async(request)


        rclpy.spin_until_future_complete(self, future, timeout_sec=10)

        if not future.done():
            raise Error("Timeout reached when calling change_gripper service")

        result: ChangeGripper.Response
        result = future.result()

        if not result.success:
            self.get_logger().error("Error calling change gripper service")
        
        waypoints = [build_pose(tc_pose.position.x, tc_pose.position.y,tc_pose.position.z + 0.4,
                                quaternion_from_euler(0.0,pi,0.0))]
        
        self._move_floor_robot_cartesian(waypoints, 0.3, 0.3)
    
    def submit_order(self, order_id):
        submit_order_client = self.create_client(
            SubmitOrder,
            '/ariac/submit_order'
        )
        request = SubmitOrder.Request()
        request.order_id = order_id

        # Send the request
        future = submit_order_client.call_async(request)

        # Wait for the response
        try:
            rclpy.spin_until_future_complete(self, future)
        except KeyboardInterrupt as kb_error:
            raise KeyboardInterrupt from kb_error

        # Check the response
        if future.result().success:
            self.get_logger().info(f'Submitted order')
        else:
            self.get_logger().warn('Unable to submit order')
