from time import sleep
from ament_index_python import get_package_share_directory
from moveit import MoveItPy, PlanningSceneMonitor
import rclpy
import pyassimp
import yaml
from rclpy.time import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.parameter import Parameter

from geometry_msgs.msg import PoseStamped, Pose, Point
from shape_msgs.msg import Mesh, MeshTriangle
from moveit_msgs.msg import CollisionObject
from std_msgs.msg import Header

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
    VacuumGripperControl
)

from std_srvs.srv import Trigger

from ariac_tutorials.utils import (
    multiply_pose,
    rpy_from_quaternion,
    rad_to_deg_str,
    AdvancedLogicalCameraImage,
    Order,
    KittingTask,
    CombinedTask,
    AssemblyTask,
    KittingPart
)


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

    def __init__(self):
        super().__init__('competition_interface')

        sim_time = Parameter(
            "use_sim_time",
            rclpy.Parameter.Type.BOOL,
            True
        )

        self.set_parameters([sim_time])

        # Service client for starting the competition
        self._start_competition_client = self.create_client(Trigger, '/ariac/start_competition')

        # Subscriber to the competition state topic
        self._competition_state_sub = self.create_subscription(
            CompetitionStateMsg,
            '/ariac/competition_state',
            self._competition_state_cb,
            10)
        
        # Store the state of the competition
        self._competition_state: CompetitionStateMsg = None

        # Subscriber to the break beam status topic
        self._break_beam0_sub = self.create_subscription(
            BreakBeamStatusMsg,
            '/ariac/sensors/breakbeam_0/status',
            self._breakbeam0_cb,
            qos_profile_sensor_data)
        
        # Store the number of parts that crossed the beam
        self._conveyor_part_count = 0
        
        # Store whether the beam is broken
        self._object_detected = False

        # Subscriber to the logical camera topic
        self._advanced_camera0_sub = self.create_subscription(
            AdvancedLogicalCameraImageMsg,
            '/ariac/sensors/advanced_camera_0/image',
            self._advanced_camera0_cb,
            qos_profile_sensor_data)
        
        # Store each camera image as an AdvancedLogicalCameraImage object
        self._camera_image: AdvancedLogicalCameraImage = None

        # Subscriber to the order topic
        self.orders_sub = self.create_subscription(
            OrderMsg,
            '/ariac/orders',
            self._orders_cb,
            10)
        
        # Flag for parsing incoming orders
        self._parse_incoming_order = False
        
        # List of orders
        self._orders = []
        
        # Subscriber to the floor gripper state topic
        self._floor_robot_gripper_state_sub = self.create_subscription(
            VacuumGripperState,
            '/ariac/floor_robot_gripper_state',
            self._floor_robot_gripper_state_cb,
            qos_profile_sensor_data)

        # Service client for turning on/off the vacuum gripper on the floor robot
        self._floor_gripper_enable = self.create_client(
            VacuumGripperControl,
            "/ariac/floor_robot_enable_gripper")

        # Attribute to store the current state of the floor robot gripper
        self._floor_robot_gripper_state = VacuumGripperState()

        # Moveit_py variables
        self._robot_moveit_py = MoveItPy(node_name="ariac_tutorials")
        self._robot_moveit_py_state = RobotState(self._robot_moveit_py.get_robot_model())

        self._floor_robot = self._robot_moveit_py.get_planning_component("floor_robot")
        self._ceiling_robot = self._robot_moveit_py.get_planning_component("ceiling_robot")

        self._planning_scene_monitor = self._robot_moveit_py.get_planning_scene_monitor()


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

    def _orders_cb(self, msg: Order):
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
            try:
                rclpy.spin_once(self)
            except KeyboardInterrupt:
                return

        self.get_logger().info('Competition is ready. Starting...')

        # Check if service is available
        if not self._start_competition_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error('Service \'/ariac/start_competition\' is not available.')
            return

        # Create trigger request and call starter service
        request = Trigger.Request()
        future = self._start_competition_client.call_async(request)

        # Wait until the service call is completed
        rclpy.spin_until_future_complete(self, future)

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
    
    def _call_get_cartesian_path (self, 
                                 waypoints : list,
                                 max_step : float):

        self.get_logger().info("Getting cartesian path")

        request = GetCartesianPath.Request()

        header = Header()
        header.frame_id = "world"
        header.stamp = self.get_clock().now().to_msg()

        request.header = header
        request.start_state = robotStateToRobotStateMsg(self._robot_moveit_py_state.update())
        request.group_name = "floor_robot"
        request.link_name = "floor_gripper"
        request.waypoints = waypoints
        request.max_step = max_step

        future = self.get_cartesian_path_client.call_async(request)


        rclpy.spin_until_future_complete(self, future, timeout_sec=10)

        if not future.done():
            raise Error("Timeout reached when calling move_cartesian service")

        result: GetCartesianPath.Response
        result = future.result()

        return result.solution

    def _call_get_position_fk (self):

        self.get_logger().info("Getting cartesian path")

        request = GetPositionFK.Request()

        header = Header()
        header.frame_id = "world"
        header.stamp = self.get_clock().now().to_msg()
        request.header = header

        request.fk_link_names = ["floor_gripper"]

        request.robot_state = robotStateToRobotStateMsg(self._robot_moveit_py_state.update())

        future = self.get_position_fk_client.call_async(request)


        rclpy.spin_until_future_complete(self, future, timeout_sec=10)

        if not future.done():
            raise Error("Timeout reached when calling move_cartesian service")

        result: GetPositionFK.Response
        result = future.result()

        return result.pose_stamped
    
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
        self._floor_robot.set_stat_state_to_current_state()
        self._floor_robot.set_goal_state(confiuration_name="home")
        self._plan_and_execute(self._robot_moveit_py,self._floor_robot, self.get_logger(), sleep_time=0.0)
        
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
    
        package_share_directory = get_package_share_directory("test_competitor")
        model_path = package_share_directory + "/meshes/"+mesh_file
        collision_object = self._makeMesh(name, model_pose,model_path, self.get_logger())
        with self._planning_scene_monitor.read_write() as scene:
            scene.apply_collision_object(collision_object)
            scene.current_state.update()
    
    def add_objects_to_planning_scene(self):
        print("HOLD")
        with open('collision_object_info.yaml','r') as object_file:
            objects_dict = yaml.safe_load(object_file)
        
        objects_dict : dict
        for key in objects_dict.keys():

            object_pose = Pose()
            
            object_pose.position.x = objects_dict[key]["position"][0]
            object_pose.position.y = objects_dict[key]["position"][1]
            object_pose.position.z = objects_dict[key]["position"][2]
            
            object_pose.orientation.x = objects_dict[key]["orientation"][0]
            object_pose.orientation.y = objects_dict[key]["orientation"][1]
            object_pose.orientation.z = objects_dict[key]["orientation"][2]
            object_pose.orientation.w = objects_dict[key]["orientation"][3]

            self._add_model_to_planning_scene(key, objects_dict[key]["file"], object_pose)
    
    def  floor_robot_pick_bin_part(part : PartMsg):
        print("HOLD")
        '''
        read the camera and see if the part is found
        If the part is found, get the pose of the part and move directly above the part using:
        
        self._move_to_cartesian_pose(part_pose)

        
        
        Then use use the cartesian path service to move down to the part
        
        robot_trajectory = self._call_get_cartesian_path_service_floor_robot(waypoints, max_step)
        self._robot_moveit_py.execute(robot_trajectory, controllers=[])

        
        
        Then attempt to pick up the parts
        
        FloorRobotSetGripperState(true);
        FloorRobotWaitForAttach(3.0);
        

        Move the robot up

        robot_trajectory = self._call_get_cartesian_path_service_floor_robot(waypoints, max_step)
        '''