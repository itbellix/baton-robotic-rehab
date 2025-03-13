import struct
import time
import threading
from typing import NamedTuple

import serial
from crc import Calculator, Configuration
import numpy as np

import rospy
from std_msgs.msg import Float32MultiArray

from experiment_parameters import *


class Config(NamedTuple):
    baud_rate: int
    sinc_length: int
    chop_enable: int
    fast_enable: int
    fir_disable: int
    temp_compensation: int  # 0: Disabled (recommended), 1: Enabled
    use_calibration: int  # 1: calibration matrix active, 0: raw measurements
    data_format: int  # 0: binary, 1: CSV
    baud_rate_config: int  # 0: 9600, 1: 57600, 2: 115200, 3: 230400, 4: 460800
    frame_header: bytes

    @staticmethod
    def default():
        return Config(
            baud_rate=460800,
            sinc_length=57,
            chop_enable=1,
            fast_enable=1,
            fir_disable=1,
            temp_compensation=0,
            use_calibration=1,
            data_format=0,
            baud_rate_config=4,
            frame_header=b"\xaa",
        )


class Reading(NamedTuple):
    fx: float
    fy: float
    fz: float
    mx: float
    my: float
    mz: float
    timestamp: float
    temperature: float

    @staticmethod
    def zero():
        return Reading(0, 0, 0, 0, 0, 0, 0, 0)

    def __sub__(self, other):
        return Reading(
            fx=self.fx - other.fx,
            fy=self.fy - other.fy,
            fz=self.fz - other.fz,
            mx=self.mx - other.mx,
            my=self.my - other.my,
            mz=self.mz - other.mz,
            timestamp=self.timestamp,
            temperature=self.temperature,
        )

    def to_dict(self):
        return {
            "fx": self.fx,
            "fy": self.fy,
            "fz": self.fz,
            "mx": self.mx,
            "my": self.my,
            "mz": self.mz,
            "time": self.timestamp,
            "temperature": self.temperature,
        }


class BotaSerialSensor:
    def __init__(self, port: str, config: Config = Config.default()):
        self._ser = serial.Serial()
        self._pd_thread_stop_event = threading.Event()
        self._status = None
        self.reading = Reading.zero()
        self.header = config.frame_header
        self._ser.baudrate = config.baud_rate
        self._ser.port = port
        self._ser.timeout = 10
        self.reading_pub = rospy.Publisher(shared_ros_topics['ft_sensor_data'], Float32MultiArray, queue_size=10)
        self.reading_message = Float32MultiArray()
        self.reading_message.data = np.zeros(8)
        self.reading_message.layout.data_offset = 0

        try:
            self._ser.open()
            rospy.loginfo("[BotaSenseOne]Opened serial port {}".format(port))
        except:
            raise Exception("[BotaSenseOne]Could not open port")

        if not self._ser.is_open:
            raise Exception("[BotaSenseOne]Could not open port")

        # configuration
        if not self.setup(config):
            rospy.loginfo("[BotaSenseOne]Failed to setup sensor")
            return
        rospy.loginfo("[BotaSenseOne]Sensor setup complete")

    def setup(self, config: Config):
        # Wait for streaming of data
        out = self._ser.read_until(bytes("App Init", "ascii"))
        if not self.contains_bytes(bytes("App Init", "ascii"), out):
            rospy.loginfo("[BotaSenseOne]Sensor not streaming, check if correct port selected!")
            return False
        time.sleep(0.5)
        self._ser.reset_input_buffer()
        self._ser.reset_output_buffer()

        # Go to CONFIG mode
        cmd = bytes("C", "ascii")
        self._ser.write(cmd)
        out = self._ser.read_until(bytes("r,0,C,0", "ascii"))
        if not self.contains_bytes(bytes("r,0,C,0", "ascii"), out):
            rospy.loginfo("[BotaSenseOne]Failed to go to CONFIG mode.")
            return False

        # Communication setup
        comm_setup = f"c,{config.temp_compensation},{config.use_calibration},{config.data_format},{config.baud_rate_config}"
        cmd = bytes(comm_setup, "ascii")
        self._ser.write(cmd)
        out = self._ser.read_until(bytes("r,0,c,0", "ascii"))
        if not self.contains_bytes(bytes("r,0,c,0", "ascii"), out):
            rospy.loginfo("[BotaSenseOne]Failed to set communication setup.")
            return False
        self.time_step = 0.00001953125 * config.sinc_length
        rospy.loginfo("[BotaSenseOne]Timestep: {}".format(self.time_step))

        # Filter setup
        filter_setup = f"f,{config.sinc_length},{config.chop_enable},{config.fast_enable},{config.fir_disable}"
        cmd = bytes(filter_setup, "ascii")
        self._ser.write(cmd)
        out = self._ser.read_until(bytes("r,0,f,0", "ascii"))
        if not self.contains_bytes(bytes("r,0,f,0", "ascii"), out):
            rospy.loginfo("[BotaSenseOne]Failed to set filter setup.")
            return False

        # Go to RUN mode
        cmd = bytes("R", "ascii")
        self._ser.write(cmd)
        out = self._ser.read_until(bytes("r,0,R,0", "ascii"))
        if not self.contains_bytes(bytes("r,0,R,0", "ascii"), out):
            rospy.loginfo("[BotaSenseOne]Failed to go to RUN mode.")
            return False

        return True

    def contains_bytes(self, subsequence, sequence):
        return subsequence in sequence

    def _processdata_thread(self):
        while not self._pd_thread_stop_event.is_set():
            frame_synced = False
            crc16X25Configuration = Configuration(
                16, 0x1021, 0xFFFF, 0xFFFF, True, True
            )
            crc_calculator = Calculator(crc16X25Configuration)

            while not frame_synced and not self._pd_thread_stop_event.is_set():
                possible_header = self._ser.read(1)
                if self.header == possible_header:
                    data_frame = self._ser.read(34)
                    crc16_ccitt_frame = self._ser.read(2)

                    crc16_ccitt = struct.unpack_from("H", crc16_ccitt_frame, 0)[0]
                    checksum = crc_calculator.checksum(data_frame)
                    if checksum == crc16_ccitt:
                        rospy.loginfo("[BotaSenseOne]Frame synced")
                        frame_synced = True
                    else:
                        self._ser.read(1)

            while frame_synced and not self._pd_thread_stop_event.is_set():
                frame_header = self._ser.read(1)

                if frame_header != self.header:
                    rospy.loginfo("[BotaSenseOne]Lost sync")
                    frame_synced = False
                    break

                data_frame = self._ser.read(34)
                crc16_ccitt_frame = self._ser.read(2)

                crc16_ccitt = struct.unpack_from("H", crc16_ccitt_frame, 0)[0]
                checksum = crc_calculator.checksum(data_frame)
                if checksum != crc16_ccitt:
                    rospy.loginfo("[BotaSenseOne]CRC mismatch received")
                    break

                self._status = struct.unpack_from("H", data_frame, 0)[0]
                self.reading = Reading(
                    fx=struct.unpack_from("f", data_frame, 2)[0],
                    fy=struct.unpack_from("f", data_frame, 6)[0],
                    fz=struct.unpack_from("f", data_frame, 10)[0],
                    mx=struct.unpack_from("f", data_frame, 14)[0],
                    my=struct.unpack_from("f", data_frame, 18)[0],
                    mz=struct.unpack_from("f", data_frame, 22)[0],
                    timestamp=struct.unpack_from("I", data_frame, 26)[0] * 1e-6,
                    temperature=struct.unpack_from("f", data_frame, 30)[0],
                )

                if Reading.timestamp != self.reading_message.data[6]:
                    self.reading_message.data = np.array([self.reading.fx, self.reading.fy, self.reading.fz,
                                                           self.reading.mx, self.reading.my, self.reading.mz,
                                                           self.reading.timestamp, self.reading.temperature])
                    self.reading_pub.publish(self.reading_message)

    def start(self):
        self.proc_thread = threading.Thread(target=self._processdata_thread)
        self.proc_thread.start()

    def close(self):
        self._pd_thread_stop_event.set()
        self.proc_thread.join()
        self._ser.close()
