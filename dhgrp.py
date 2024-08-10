import pymodbus.client as ModbusClient
from time import sleep
from ctypes import c_uint16

import numpy as np

from system import System, InputPort, OutputPort, EventSystem


class DHGripper(EventSystem):
    def __init__(self, port: str):
        super().__init__(inputs=['grip', 'force', 'speed'])
        self.client = ModbusClient.ModbusSerialClient(
            port=port,
            baudrate=115200,
            bytesize=8,
            parity="N",
            stopbits=1,
        )

    def _state_g(self):
        return self.client.read_holding_registers(0x200, 1, slave=1).registers[0]

    def _state_r(self):
        return self.client.read_holding_registers(0x20A, 1, slave=1).registers[0]

    def on_start(self):
        if self._state_g() != 1 or self._state_r() != 1:
            self.client.write_register(0x100, 0xa5, slave=1)
            while self._state_g() != 1 and self._state_r() != 1:
                sleep(0.1)

    @EventSystem.on_event('grip')
    def on_grip(self, value):
        self.client.write_register(0x103, c_uint16(value).value, slave=1)

    @EventSystem.on_event('force')
    def on_force(self, value):
        self.client.write_register(0x101, c_uint16(value).value, slave=1)

    @EventSystem.on_event('speed')
    def on_speed(self, value):
        self.client.write_register(0x104, c_uint16(value).value, slave=1)

# connection = client.connect()

# slave = 1

# client = ModbusClient.ModbusSerialClient(
#             port='/dev/ttyUSB0',
#             baudrate=115200,
#             bytesize=8,
#             parity="N",
#             stopbits=1,
#         )

def wait_stop_g():
    while (r := client.read_holding_registers(0x201, 1, slave=1).registers[0]) == 0:
        sleep(0.1)
    return r

def wait_stop_r():
    while (r := client.read_holding_registers(0x20B, 1, slave=1).registers[0]) == 0:
        sleep(0.1)
    return r

def rotate(a, sync = True):
    client.write_register(0x105, c_uint16(a).value, slave=1)
    if sync:
        return wait_stop_r()
    return None

def set_rot_force(f):
    client.write_register(0x108, c_uint16(f).value, slave=1)

def set_force(f):
    client.write_register(0x101, c_uint16(f).value, slave=1)

def set_rot_speed(s):
    client.write_register(0x107, c_uint16(s).value, slave=1)

def set_speed(s):
    client.write_register(0x104, c_uint16(s).value, slave=1)

def grip(a, sync = True):
    client.write_register(0x103, c_uint16(a).value, slave=1)
    if sync:
        return wait_stop_g()
    return None

def state_g():
    return client.read_holding_registers(0x200, 1, slave=1).registers[0]

def state_r():
    return client.read_holding_registers(0x20A, 1, slave=1).registers[0]

def init():
    if state_g() != 1 or state_r() != 1:
        client.write_register(0x100, 0xa5, slave=1)
        while state_g() != 1 and state_r() != 1:
            sleep(0.1)

def get_test_io_params():
    return client.read_holding_registers(0x400, 1, slave=1).registers[0]

def set_test_io_params(param: int):
    client.write_register(0x400, c_uint16(param).value, slave=1)

def get_io_mode():
    return client.read_holding_registers(0x402, 1, slave=1).registers[0]

def set_io_mode(mode: bool):
    client.write_register(0x402, c_uint16(mode).value, slave=1)

def get_io_params(grp: int):
    pos = client.read_holding_registers(0x405 + grp * 3, 1, slave=1).registers[0]
    frc = client.read_holding_registers(0x406 + grp * 3, 1, slave=1).registers[0]
    return (pos, frc)

def set_io_params(grp: int, pos: int, frc: int):
    client.write_register(0x405 + grp * 3, c_uint16(pos).value, slave=1)
    client.write_register(0x406 + grp * 3, c_uint16(frc).value, slave=1)

def save_params():
    client.write_register(0x300, c_uint16(1).value, slave=1)

def write_hold_reg(add: int, val: int):
    client.write_register(add, c_uint16(val).value, slave=slave)

def read_hold_reg(add: int):
    return client.read_holding_registers(add, 1, slave=slave).registers

def read_in_reg(add: int):
    return client.read_input_registers(add, 1, slave=slave).registers

def write_coil(add: int, val: int):
    client.write_coil(add, c_uint16(val).value, slave=slave)

def read_coil(add: int, len: int = 1):
    return client.read_coils(add, len, slave=slave).bits

def read_dis_in(add: int, len: int = 1):
    return client.read_discrete_inputs(add, len, slave=slave).bits

def test():
    init()

    set_speed(50)
    set_force(100)

    while True:
        for v in range(0, 1000, 333):
            grip(v)
            wait_stop_g()
        grip(0)


if __name__ == "__main__":
    # test()
    gripper = DHGripper("/dev/ttyUSB0")
    gripper.start()
    gripper.ins.speed.write(20)
    gripper.ins.force.write(100)

    try:
        while True:
            for width in (np.cos(np.linspace(0, 2 * np.pi, 15)) + 1) * 500:
                gripper.ins.grip.write(int(width))
                print(int(width))
                sleep(0.02)
    except KeyboardInterrupt:
        gripper.stop()


    # while True:
    #     for v in range(0, 1000, 333):
    #         runner.send(("grip", 0, v))
