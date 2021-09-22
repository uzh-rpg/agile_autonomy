import os
import time
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Empty, Float32
from nav_msgs.msg import Odometry
import rospy
import numpy as np
from pyquaternion import Quaternion


class MessageHandler():
    def __init__(self):
        self.autopilot_off = rospy.Publisher("/hummingbird/autopilot/off", Empty,
                                             queue_size=1)
        self.arm_bridge = rospy.Publisher("/hummingbird/bridge/arm", Bool,
                                          queue_size=1)
        self.autopilot_start = rospy.Publisher("/hummingbird/autopilot/start", Empty,
                                               queue_size=1)
        self.autopilot_pose_cmd = rospy.Publisher("/hummingbird/autopilot/pose_command",
                                                  PoseStamped, queue_size=1)
        self.tree_spacing_cmd = rospy.Publisher("/hummingbird/tree_spacing",
                                                Float32, queue_size=1)
        self.obj_spacing_cmd = rospy.Publisher("/hummingbird/object_spacing",
                                               Float32, queue_size=1)
        self.reset_exp_pub = rospy.Publisher("/success_reset",
                                             Empty, queue_size=1)
        self.save_pc_pub = rospy.Publisher("/hummingbird/save_pc",
                                           Bool, queue_size=1)

    def publish_autopilot_off(self):
        msg = Empty()
        self.autopilot_off.publish(msg)
        time.sleep(1)

    def publish_reset(self):
        msg = Empty()
        self.reset_exp_pub.publish(msg)

    def publish_tree_spacing(self, spacing):
        msg = Float32()
        msg.data = spacing
        print("Setting Tree Spacing to {}".format(msg.data))
        self.tree_spacing_cmd.publish(msg)

    def publish_obj_spacing(self, spacing):
        msg = Float32()
        msg.data = spacing
        print("Setting Object Spacing to {}".format(msg.data))
        self.obj_spacing_cmd.publish(msg)

    def publish_arm_bridge(self):
        msg = Bool()
        msg.data = True
        self.arm_bridge.publish(msg)

    def publish_save_pc(self):
        msg = Bool()
        msg.data = True
        self.save_pc_pub.publish(msg)

    def publish_autopilot_start(self):
        msg = Empty()
        self.autopilot_start.publish(msg)
        time.sleep(25)

    def publish_goto_pose(self, pose=[0, 0, 3.]):
        msg = PoseStamped()
        msg.header.frame_id = "world"
        msg.pose.position.x = pose[0]
        msg.pose.position.y = pose[1]
        msg.pose.position.z = pose[2]
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0
        self.autopilot_pose_cmd.publish(msg)
        time.sleep(3)

def setup_sim(msg_handler, config):
    print("==========================")
    print("     RESET SIMULATION     ")
    print("==========================")

    # after this message, autopilot will automatically go to 'BREAKING' and 'HOVER' state since
    # no control_command_inputs are published any more
    os.system("rosservice call /gazebo/pause_physics")
    print("Unpausing Physics...")
    os.system("rosservice call /gazebo/unpause_physics")
    print("Placing quadrotor...")
    msg_handler.publish_autopilot_off()
    # get a position
    pos_choice = np.random.choice(len(config.unity_start_pos))
    position = np.array(config.unity_start_pos[pos_choice])
    # No yawing possible for trajectory generation
    start_quaternion = Quaternion(axis=[0,0,1], angle=position[-1]).elements

    start_string = "rosservice call /gazebo/set_model_state " + \
     "'{model_state: { model_name: hummingbird, pose: { position: { x: %f, y: %f ,z: %f }, " % (position[0],position[1],position[2]) + \
     "orientation: {x: %f, y: %f, z: %f, w: %f}}, " % (start_quaternion[1],start_quaternion[2],start_quaternion[3],start_quaternion[0]) + \
     "twist:{ linear: {x: 0.0 , y: 0 ,z: 0 } , angular: { x: 0.0 , y: 0 , z: 0.0 }}, " + \
     "reference_frame: world } }'"

    os.system(start_string)
    return position


def place_quad_at_start(msg_handler):
    '''
    start position: a tuple, array, or list with [x,y,z] representing the start position.
    '''
    # Make sure to use GT odometry in this step
    msg_handler.publish_autopilot_off()
    # reset quad to initial position
    msg_handler.publish_arm_bridge()
    msg_handler.publish_autopilot_start()
    return
