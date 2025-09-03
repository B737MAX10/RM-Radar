# rx 0x020c 0x020E  tx 0x0305
import serial
from crc import *
import struct
import serial.tools.list_ports
import binascii


team = 'R'

SOF = 0xA5
map_robot_data_t_cmd_id = 0x0305
Dart_info = 0x0105


class RefereeInfo:
    def __init__(self) -> None:
        self.dart_info = {}
        self.count = 0
        self.mark_data = {}
        self.double_state = 0
        self.result = 0


referee_info = RefereeInfo()


class SerialPort:
    def __init__(self, COM):
        self.ser = serial.Serial(
            port=COM,
            baudrate=115200,
            bytesize=serial.EIGHTBITS,
            stopbits=serial.STOPBITS_ONE,
            parity=serial.PARITY_NONE,
            timeout=None,
            write_timeout=None
        )
        if self.ser.isOpen():
            print("串口打开成功。")
        else:
            print("串口打开失败。")

        self.SEQ = [int(0)]
        self.send_count = [0]

    def close(self):
        if self.ser.isOpen():
            self.ser.close()
            print("串口关闭成功。")
        else:
            print("串口关闭失败。")

    def tx(self, data):
        if self.ser.isOpen():
            message = bytes([SOF, 0x0A, 0x00]) + self.SEQ[0].to_bytes(1, byteorder="little")  # SOF=0xA5, data_length=0x0A00(高位在后，低位在前)
            message += append_crc8(message[:4])
            message += map_robot_data_t_cmd_id.to_bytes(2, byteorder="little")  # map_robot_data_t cmd_id=0x0305
            message += data["ID"].to_bytes(2, byteorder="little")
            message += struct.pack("f", data["position"][0])
            message += struct.pack("f", data["position"][1])
            message += append_crc16(message)
            print_bytes(message)
            self.SEQ[0] += 1
            if self.SEQ[0] > 0xFF:
                self.SEQ[0] = 0
            self.ser.write(message)
        else:
            print("串口未打开。")

    def tx_double(self):
        if self.ser.isOpen():
            if team == 'R':
                id = 0x9
            elif team == 'B':
                id = 0x109
            message = bytes([SOF, 0x07, 0x00]) + self.SEQ[0].to_bytes(1, byteorder="little")
            message += append_crc8(message[:4])
            message += bytes([0x01, 0x03])
            if referee_info.double_state != 1 and referee_info.count > 0:
                self.send_count[0] += 1
                if self.send_count[0] == 1:
                    message += struct.pack("<HHHB", 0x0121, id, 0x8080, 1)
                elif self.send_count[0] >= 2:
                    message += struct.pack("<HHHB", 0x0121, id, 0x8080, 2)
            else:
                message += struct.pack("<HHHB", 0x0121, id, 0x8080, 0)
            message += append_crc16(message)
            self.SEQ[0] += 1
            if self.SEQ[0] > 0xFF:
                self.SEQ[0] = 0
            self.ser.write(message)
            print("send double data")
        else:
            print("串口未打开。")

    def rx(self):
        while self.ser.isOpen():
            if self.ser.read(1) == b"\xA5":
                frame_header = bytes([0xA5])
                frame_header += self.ser.read(4)
                if verify_crc8(frame_header):
                    data_length = struct.unpack("<H", frame_header[1:3])
                    data = frame_header + self.ser.read(data_length[0] + 4)
                    if verify_crc16(data):
                        cmd_id = struct.unpack("<H", data[5:7])
                        if cmd_id[0] == 0x020C:
                            referee_info.mark_data["hero"] = data[7]
                            referee_info.mark_data["engineer"] = data[8]
                            referee_info.mark_data["standard1"] = data[9]
                            referee_info.mark_data["standard2"] = data[10]
                            referee_info.mark_data["standard3"] = data[11]
                            referee_info.mark_data["sentry"] = data[12]
                        elif cmd_id[0] == 0x020E:
                            referee_info.count = data[7] & 0b11
                            referee_info.double_state = data[7] & 0b100
                            print("count: ", referee_info.count)
                            print("double state: ", referee_info.double_state)
                        elif cmd_id[0] == Dart_info:
                            referee_info.dart_info['flag'] = data[8]
                            flag = referee_info.dart_info['flag']
                            referee_info.result = (flag & 0b1100000) >> 5
                        else:
                            print("未知命令。")
        else:
            print("串口未打开。")


# serial_port = SerialPort("COM7")  # 根据连接的 COM 修改端口号,仅修改数字

def list_available_ports():  # 不知道什么端口能用就调用这个，会打印当前可用端口，ttl转usb的那个COM口会有“CH340”
    ports = serial.tools.list_ports.comports()
    for port in ports:
        print(port)

def print_bytes(data):
    hex_data = binascii.hexlify(data)
    hex_str = hex_data.decode("utf-8")
    hex_str_with_spaces = " ".join(hex_str[i : i + 2] for i in range(0, len(hex_str), 2))
    print("Bytes: " + hex_str_with_spaces)
