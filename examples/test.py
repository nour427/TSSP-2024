import sys
import time
import logging
import rtde.rtde as rtde
import rtde.rtde_config as rtde_config

sys.path.append("..")

logging.basicConfig(level=logging.DEBUG)

ROBOT_HOST = "192.168.0.3"
ROBOT_PORT = 30004
config_filename = "control_loop_configuration.xml"

keep_running = True

logging.getLogger().setLevel(logging.INFO)
conf = rtde_config.ConfigFile(config_filename)
state_names, state_types = conf.get_recipe("state")
setp_names, setp_types = conf.get_recipe("setp")
watchdog_names, watchdog_types = conf.get_recipe("watchdog")

con = rtde.RTDE(ROBOT_HOST, ROBOT_PORT)
con.connect()

test = con.is_connected()
print("is_connected: ", test)
con.send_start()
if test:
    #if connection_state:
    print('Connected to the robot')
else:
    print('Failed to connect to the robot')
    sys.exit()

# get controller version
con.get_controller_version()
print('setp_names', setp_names)
print('setp types', setp_types)

# setup recipes
con.send_output_setup(state_names, state_types)

try:
    setp = con.send_input_setup(setp_names, setp_types)
    watchdog = con.send_input_setup(watchdog_names, watchdog_types)
except Exception as e:
    logging.error(f'Error setting up inputs: {e}')

con.send_start()


con.send_start()


# Setpoints to move the robot to
setp1 = [0.4, -0.21, 0.05, 0, 3.11, 0.04]
setp2 = [0.68, -0.21, 0.21, 0, 3.11, 0.04]
# setp3 = [0.4, 0.18, 0.05, 0, 3.11, 0.04]
# setp4 = [0.4, 0.18, 0.05, 0, 3.11, 0.04]

setp.input_double_register_0 = 0
setp.input_double_register_1 = 0
setp.input_double_register_2 = 0
setp.input_double_register_3 = 0
setp.input_double_register_4 = 0
setp.input_double_register_5 = 0

# The function "rtde_set_watchdog" in the "rtde_control_loop.urp" creates a 1 Hz watchdog
watchdog.input_int_register_0 = 0


def setp_to_list(sp):
    sp_list = []
    for i in range(0, 6):
        sp_list.append(sp.__dict__["input_double_register_%i" % i])
    return sp_list


def list_to_setp(sp, list):
    for i in range(0, 6):
        sp.__dict__["input_double_register_%i" % i] = list[i]
    return sp


# start data synchronization
if not con.send_start():
    sys.exit()

# control loop
move_completed = True
while keep_running:
    # receive the current state
    state = con.receive()
    if state is None:
        print("Failed to receive state")
        break

    pose = state.actual_TCP_pose
    setp3 = [0.500, 0, 0.050, -2.25292, -2.1894, -0.0000]

    list_to_setp(setp, setp3)
    con.send(setp)
    # print(pose)

    # kick watchdog
    con.send(watchdog)

con.send_pause()
con.disconnect()
