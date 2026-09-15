## ===========================
## nick's code, 2026-02-27
## lkmotor stable library
## ===========================

import can
import time
import platform
from typing import Optional


def open_bus(bitrate: int = 1_000_000, **kwargs) -> "can.BusABC":
    """Open a single CAN bus appropriate for the current platform.

    - macOS: a USB-CAN adapter via a userspace backend (no SocketCAN, no
      ``sudo``). The adapter is auto-detected by USB id (see ``mac_can``):
        * PEAK PCAN-USB       -> ``pcan`` backend (MacCAN ``libPCBUSB.dylib``)
        * CANable/candleLight -> ``gs_usb`` backend (libusb)
    - Linux: SocketCAN ``can0`` (bring it up first with
      ``sudo ip link set can0 up type can bitrate 1000000``).

    A USB adapter can only be opened once, so share the returned bus across all
    motors (pass it as ``LKMotor(motor_id=.., bus=bus)``).
    """
    if platform.system().lower() == "darwin":
        import mac_can

        adapter = mac_can.detect_adapter()

        if adapter == "pcan":
            # PEAK PCAN-USB via the MacCAN libPCBUSB.dylib userspace driver.
            mac_can.ensure_pcbusb_loadable()
            return can.Bus(
                interface="pcan", channel="PCAN_USBBUS1",
                bitrate=bitrate, **kwargs,
            )

        if adapter == "gs_usb":
            mac_can.apply_mac_gs_usb_patches()
            # The first open of a freshly-enumerated adapter is reliable. A few
            # short retries cover a transiently-busy adapter (still re-enumerating
            # right after a previous process). We deliberately keep this SHORT:
            # each attempt calls gs_usb.start() -> USB reset(), and hammering an
            # already wedged adapter only deepens the wedge, so fail fast.
            import gc

            attempts = 3
            for i in range(attempts):
                try:
                    return can.Bus(
                        interface="gs_usb", channel="canable", index=0,
                        bitrate=bitrate, **kwargs,
                    )
                except Exception as exc:
                    gc.collect()  # release the half-claimed libusb handle
                    if i == attempts - 1:
                        raise type(exc)(
                            f"{exc}\n  gs_usb adapter open failed after {attempts} tries. "
                            "It may be wedged -- unplug the CANable, wait ~10 s, replug "
                            "directly into the Mac, then retry. Avoid opening the bus in "
                            "two processes back-to-back."
                        ) from exc
                    time.sleep(0.4)

        raise can.CanInitializationError(
            "No supported USB-CAN adapter found on macOS. Plug in a PEAK "
            "PCAN-USB or a CANable/candleLight (or check the USB cable). "
            "Looked for PEAK 0c72:000c/0012 and candleLight 1d50:606f."
        )

    return can.Bus(
        interface="socketcan", channel="can0", bitrate=bitrate, **kwargs
    )


class LKMotor:
    def __init__(
            self,
            motor_id: int,
            bus: Optional["can.BusABC"] = None,
            bus_interface: Optional[str] = None,
            bus_channel: Optional[str] = None,
            bitrate: int = 1_000_000,
            **kwargs,
    ) -> None:
        """
        Initialize the motor.

        Args:
            motor_id (int): Motor ID
            bus (can.BusABC, optional): An already-open CAN bus to share. A
                ``gs_usb`` adapter can only be opened once, so on macOS all
                motors on the same adapter MUST share one bus. When given, the
                motor will not shut the bus down on ``motor_release()``.
            bus_interface (str, optional): CAN bus interface, e.g. "socketcan".
                Honoured on Linux; ignored on macOS (always ``gs_usb``).
            bus_channel (str, optional): CAN bus channel, e.g. "can0".
            bitrate (int): CAN bitrate in bits/s (default 1 Mbit/s).
            **kwargs: Additional arguments forwarded to the bus backend.
        """

        self.motor_id = motor_id
        if bus is not None:
            self.bus = bus
            self._owns_bus = False
        elif bus_interface is not None and platform.system().lower() != "darwin":
            self.bus = can.interface.Bus(
                interface=bus_interface, channel=bus_channel, bitrate=bitrate, **kwargs
            )
            self._owns_bus = True
        else:
            # macOS, or no interface specified: pick the platform default.
            self.bus = open_bus(bitrate=bitrate, **kwargs)
            self._owns_bus = True

        self.temperature = 0
        self.voltage = 0
        self.current = 0
        self.motor_state = 0
        self.error_state = 0
        self.iq = 0
        self.speed = 0
        self.multi_turn_angle = 0
        self.single_turn_angle = 0
        self.encoder_value = 0
        self.encoder_raw = 0
        self.encoder_offset = 0
        self.current_A = 0
        self.current_B = 0
        self.current_C = 0

    def _decimal_to_byte(self, value, digit) -> list:
        """
        Convert a decimal number to a byte list.

        Args:
            value(int): Decimal number
            digit(int): Number of bytes

        Returns:
            list: Byte list, arranged from low to high
        """

        byte_list = []
        if value < 0:
            value -= 1 << digit * 8
        for i in range(digit):
            byte_list.append((value >> 8 * i) & 0xFF)

        return byte_list

    def _byte_to_decimal(self, values: list) -> int:
        """
        Convert a byte list to a decimal number.

        Args:
            values(list): Byte list, arranged from low to high

        Returns:
            int: Decimal number
        """

        value_dec = 0
        for i in range(len(values)):
            value_dec += values[i] << 8 * i
        if values[-1] >> 7 == 1:
            value_dec -= 1 << len(values) * 8

        return value_dec

    def _send_command(self, command_byte, data=None) -> None:
        """
        Send a command to the motor.

        Args:
            command_byte (uint8_t): Command byte
            data (unit8_t list): Data bytes to be sent
        """

        data = [command_byte] + (data if data else [0x00] * 7)
        message = can.Message(
            arbitration_id=0x140 + self.motor_id, data=data, is_extended_id=False
        )

        self.bus.send(message)

    def flush_rx(self) -> int:
        """
        Drop every CAN frame currently waiting in the RX queue.

        Fire-and-forget streams (e.g. send_zero_command during power-on) leave the
        motor's unread reply frames piling up in the adapter/OS buffer. A later
        blocking read returns the OLDEST queued frame, so telemetry can lag the
        motor by the whole backlog (seconds to a minute). Call this before
        switching from a fire-and-forget stream to reply-reading control so reads
        start on live data. Returns the number of frames discarded.
        """

        dropped = 0
        while self.bus.recv(timeout=0.0) is not None:
            dropped += 1
        return dropped

    def _receive_response(self, timeout: float = 0.5) -> Optional[list]:

        expected_id = 0x140 + self.motor_id  # Motor reply ID

        start_time = time.time()
        while (time.time() - start_time) < timeout:
            response = self.bus.recv(timeout=0.05)
            if response is not None and response.is_rx:
                # is_rx filters out our own TX frames, which gs_usb loops back
                # with the same arbitration id as the motor's reply.
                if response.arbitration_id == expected_id:
                    return response.data

        return None

    def _parse_response_1(self, response: list) -> tuple:
        """
        Parse the response data from the motor.

        This function updates:
        - Motor temperature (int8_t, 1 °C/LSB)
        - Motor voltage (int16_t, 0.01 V/LSB)
        - Motor current (int16_t, 0.01 A/LSB)
        - Motor state (uint8_t)
            | Byte | Description |
            |------|-------------|
            | 0x00 | Opened      |
            | 0x10 | Closed      |
        - Error state (uint8_t)
            | Bit |     Description    | 0 |              1                |
            |-----|--------------------|---|-------------------------------|
            |  0  | Low voltage        | 0 | Low voltage protection        |
            |  1  | High voltage       | 0 | High voltage protection       |
            |  2  | Driver temperature | 0 | Driver temperature over limit |
            |  3  | Motor temperature  | 0 | Motor temperature over limit  |
            |  4  | Current            | 0 | Over current                  |
            |  5  | Short circuit      | 0 | Short circuit                 |
            |  6  | Stall              | 0 | Stall                         |
            |  7  | Input signal       | 0 | Input signal timeout          |

        Args:
            response (list): Response data from the motor
        """

        if response:
            self.temperature = self._byte_to_decimal([response[1]])
            self.voltage = self._byte_to_decimal(response[2:4]) * 0.01
            self.current = self._byte_to_decimal(response[4:6]) * 0.01
            self.motor_state = response[6]
            self.error_state = response[7]
        return (
            self.temperature,
            self.voltage,
            self.current,
            self.motor_state,
            self.error_state,
        )

    def _parse_response_2(self, response: list) -> tuple:
        """
        Parse the response data from the motor.

        This function updates:
        - Motor temperature (int8_t, 1 °C/LSB)
        - Motor iq (int16_t)
            | Model |     Unit      |
            |-------|---------------|
            | MF    | 33/4096 A/LSB |
            | MG    | 66/4096 A/LSB |
        - Motor speed (int16_t, 1 dps/LSB)
        - Encoder value (uint16_t)
            | Model |   Range   |
            |-------|-----------|
            | 14bit | 0 ~ 16383 |
            | 15bit | 0 ~ 32767 |
            | 16bit | 0 ~ 65535 |

        Args:
            response (list): Response data from the motor
        """

        if response:
            self.temperature = self._byte_to_decimal([response[1]])
            self.iq = self._byte_to_decimal(response[2:4])
            self.speed = self._byte_to_decimal(response[4:6])
            self.encoder_value = self._byte_to_decimal(response[6:8])
        return self.temperature, self.iq, self.speed, self.encoder_value

    def _parse_response_3(self, response: list) -> tuple:
        """
        Parse the response data from the motor.

        This function updates:
        - Motor temperature (int8_t, 1 °C/LSB)
        - Phase A, B, C current (int16_t)
            | Model |     Unit      |
            |-------|---------------|
            | MF    | 33/4096 A/LSB |
            | MG    | 66/4096 A/LSB |

        Args:
            response (list): Response data from the motor
        """
        if response:
            self.temperature = self._byte_to_decimal([response[1]])
            self.current_A = self._byte_to_decimal(response[2:4])
            self.current_B = self._byte_to_decimal(response[4:6])
            self.current_C = self._byte_to_decimal(response[6:8])
        return self.temperature, self.current_A, self.current_B, self.current_C

    def read_motor_status_1(self) -> Optional[tuple]:
        """
        Read the motor status 1.

        This function sends a command to the motor to read the motor status including:
        - Motor temperature (int8_t, 1 °C/LSB)
        - Motor voltage (int16_t, 0.01 V/LSB)
        - Motor current (int16_t, 0.01 A/LSB)
        - Motor state (uint8_t)
        - Error state (uint8_t)
        Then the response data is parsed and the motor status is updated.
        """

        self._send_command(0x9A)
        response = self._receive_response()
        if response:
            return self._parse_response_1(response)

    def clear_error_flags(self) -> Optional[tuple]:
        """
        Clear the error flags of the motor.

        This function sends a command to the motor to clear the error flags.
        """

        self._send_command(0x9B)
        response = self._receive_response()
        if response:
            return self._parse_response_1(response)

    def read_motor_status_2(self) -> Optional[tuple]:
        """
        Read the motor status 2.

        This function sends a command to the motor to read the motor status including:
        - Motor temperature (int8_t, 1 °C/LSB)
        - Motor iq (int16_t)
        - Motor speed (int16_t, 1 dps/LSB)
        - Encoder value (uint16_t)
        Then the response data is parsed and the motor status is updated.
        """

        self._send_command(0x9C)
        response = self._receive_response()
        if response:
            return self._parse_response_2(response)

    def read_motor_status_3(self) -> Optional[tuple]:
        """
        Read the motor status 3.

        This function sends a command to the motor to read the motor status including:
        - Motor temperature (int8_t, 1 °C/LSB)
        - Phase A, B, C current (int16_t)
        Then the response data is parsed and the motor status is updated.
        """

        self._send_command(0x9D)
        response = self._receive_response()
        if response:
            return self._parse_response_3(response)

    def motor_shutdown(self) -> Optional[list]:
        """
        Shutdown the motor.

        This function sends a command to the motor to turn off the motor.
        It will clear the number of turns and previous commands.
        The LED light will shine slowly.
        The motor can receive and respond to the commands, but does not execute them.
        """

        self._send_command(0x80)
        return self._receive_response()

    def motor_run(self) -> Optional[list]:
        """
        Start the motor.

        This function sends a command to the motor to start the motor.
        The LED light will keep on.
        The motor can receive commands and execute them.
        """

        self._send_command(0x88)
        return self._receive_response()

    def ensure_running(self) -> Optional[int]:
        """
        Make sure the motor output is enabled before sending control commands.

        Reads status-1 and, if the motor is in the stopped/closed state
        (motorState 0x10), clears any error flags (0x9B) and then sends
        motor_run() (0x88) to enable the output. A stopped motor still replies to
        commands but will not execute them until it is run, so call this as a
        startup check.

        Note: on this LK firmware the running state reads 0x30 (the protocol doc
        lists it as 0x00); only 0x10 means stopped. The MCU-side comms watchdog
        latches the input-signal-timeout flag (error bit 7, 0x80) if it trips at
        power-on; that latch clears over CAN with 0x9B + 0x88 (no power cycle) --
        use motorbus.MotorBus.recover() (or arm_motors) for the non-blocking,
        multi-motor form.

        Returns:
            Optional[int]: the motorState read before any start (0x10 stopped,
                           otherwise running), or None if the motor did not reply.
        """

        status = self.read_motor_status_1()
        if status is None:
            return None
        motor_state = status[3]
        if motor_state == 0x10:
            self.clear_error_flags()
            self.motor_run()
        return motor_state

    def clear_fault_and_restart(self) -> Optional[bool]:
        """
        DEPRECATED (single-motor only) -- prefer motorbus.MotorBus.recover(),
        which runs this minimal ladder non-blocking on a shared multi-motor bus.
        The blocking reads here discard other motors' frames (_receive_response),
        so this is safe only with one motor on the bus.

        Recover the motor from a tripped input-signal-lost latch over CAN, so a
        physical power cycle is not needed. t7_recover_no_powercycle.py measured
        that the input-lost latch (error bit 7, 0x80) clears over CAN with the
        minimal sequence clear + run -- no 0x80 shutdown step. Call once at
        startup (after power-on):

            1. read status-1 (0x9A) to see the current error byte,
            2. clear error flags (0x9B),
            3. run the motor (0x88),
            4. re-read status-1 and report whether the 0x80 flag cleared.

        Returns:
            Optional[bool]: True if the input-signal-timeout flag (0x80) is clear
                            after the attempt (recovered over CAN); False if it is
                            still latched (a power cycle is required); None if the
                            motor did not reply.
        """

        if self.read_motor_status_1() is None:
            return None
        self.clear_error_flags()
        self.motor_run()
        status = self.read_motor_status_1()
        if status is None:
            return None
        error_state = status[4]
        return not (error_state & 0x80)

    def send_zero_command(self) -> None:
        """
        Fire a single zero-torque command (0xA1, iq=0) WITHOUT waiting for a reply.

        The motor stays limp (no motion); the only purpose is to feed the MCU-side
        comms watchdog, which any received command resets. Fire-and-forget (no
        blocking read) so it can be streamed fast during power-on -- see
        motorbus.MotorBus.arm() / hold() (or the legacy prime_watchdog()). [0xA1] + 7x0x00 encodes iq = 0.
        """

        self._send_command(0xA1)

    def motor_stop(self) -> Optional[list]:
        """
        Stop the motor.

        This function sends a command to the motor to stop the motor.
        The state of the motor will not be cleared.
        The motor can receive new commands and execute them.
        """

        self._send_command(0x81)
        return self._receive_response()

    def motor_release(self) -> None:
        """
        Release the motor.

        Stops the motor. Shuts the bus down only if this motor opened it;
        a shared bus (passed in via ``bus=``) is left for the caller to close.
        """

        self.motor_stop()
        if self._owns_bus:
            self.bus.shutdown()

    def brake_control(self, control_byte: int) -> Optional[int]:
        """
        Brake control command.

        This function sends a command to the motor to control the brake.

        Args:
            control_byte (uint8_t): Control byte, 0x00 for brake, 0x01 for release, 0x10 for state
        """

        data = self._decimal_to_byte(control_byte, 1) + [0x00] * 6
        self._send_command(0x8C, data)
        response = self._receive_response()
        if response:
            return self._byte_to_decimal([response[1]])

    def open_loop_control(self, power_control: int) -> Optional[tuple]:
        """
        Open loop control command.

        This function sends a command to the motor to control the motor in open loop.
        Only realized on MS series motors.

        Args:
            power_control (int16_t): Power control, range from -850 to 850
        """

        data = [0x00] * 3 + self._decimal_to_byte(power_control, 2) + [0x00] * 2
        self._send_command(0xA0, data)
        response = self._receive_response()
        if response:
            return self._parse_response_2(response)

    def torque_loop_control(self, iq_control: int) -> Optional[tuple]:
        """
        Torque loop control command.

        This function sends a command to the motor to control the motor in torque loop.
        Only realized on MF, MH, and MG series motors.
        The iq range for MF series is -16.5 ~ 16.5 A, and for MG series is -33 ~ 33 A.

        Args:
            iq_control (int16_t): Torque control, range from -2048 to 2048
        """

        data = [0x00] * 3 + self._decimal_to_byte(iq_control, 2) + [0x00] * 2
        self._send_command(0xA1, data)
        response = self._receive_response()
        if response:
            return self._parse_response_2(response)

    def speed_loop_control(self, iq_control: int, speed_control: int) -> Optional[tuple]:
        """
        Speed loop control command.

        This function sends a command to the motor to control the motor in speed loop.

        Args:
            iq_control (int16_t): Torque control, range from -2048 to 2048
            speed_control (int32_t): Speed control, unit: 0.01 dps/LSB
        """

        data = (
                [0x00]
                + self._decimal_to_byte(iq_control, 2)
                + self._decimal_to_byte(speed_control, 4)
        )
        self._send_command(0xA2, data)
        response = self._receive_response()
        if response:
            return self._parse_response_2(response)

    def multi_turn_position_control(
            self, angle_control, max_speed: Optional[int] = None
    ) -> Optional[tuple]:
        """
        Multi-turn position control command.

        This function sends a command to the motor to control the motor in multi-turn position loop.

        Args:
            angle_control (int32_t): Angle control, unit: 0.01 degree/LSB
            max_speed (uint16_t, optional): Maximum speed, unit: 1 dps/LSB
        """

        if max_speed is None:
            data = [0x00] * 3 + self._decimal_to_byte(angle_control, 4)
            self._send_command(0xA3, data)
        else:
            data = (
                    [0x00]
                    + self._decimal_to_byte(max_speed, 2)
                    + self._decimal_to_byte(angle_control, 4)
            )
            self._send_command(0xA4, data)
        response = self._receive_response()
        if response:
            return self._parse_response_2(response)

    def single_turn_position_control(
            self, spin_direction, angle_control, max_speed: Optional[int] = None
    ) -> Optional[tuple]:
        """
        Single-turn position control command.

        This function sends a command to the motor to control the motor in single-turn position loop.

        Args:
            spin_direction (uint8_t): Spin direction, 0x00 for clockwise, 0x01 for counterclockwise
            angle_control (int32_t): Angle control, unit: 0.01 degree/LSB
            max_speed (uint16_t, optional): Maximum speed, unit: 1 dps/LSB
        """

        if max_speed is None:
            data = (
                    self._decimal_to_byte(spin_direction, 1)
                    + [0x00] * 2
                    + self._decimal_to_byte(angle_control, 4)
            )
            self._send_command(0xA5, data)
        else:
            data = (
                    self._decimal_to_byte(spin_direction, 1)
                    + self._decimal_to_byte(max_speed, 2)
                    + self._decimal_to_byte(angle_control, 4)
            )
            self._send_command(0xA6, data)
        response = self._receive_response()
        if response:
            return self._parse_response_2(response)

    def incremental_position_control(
            self, angle_increment, max_speed: Optional[int] = None
    ) -> Optional[tuple]:
        """
        Incremental position control command.

        This function sends a command to the motor to control the motor in incremental position loop.

        Args:
            angle_increment (int32_t): Angle increment, unit: 0.01 degree/LSB
            max_speed (uint16_t, optional): Maximum speed, unit: 1 dps/LSB
        """

        if max_speed is None:
            data = [0x00] * 3 + self._decimal_to_byte(angle_increment, 4)
            self._send_command(0xA7, data)
        else:
            data = (
                    [0x00]
                    + self._decimal_to_byte(max_speed, 2)
                    + self._decimal_to_byte(angle_increment, 4)
            )
            self._send_command(0xA8, data)
        response = self._receive_response()
        if response:
            return self._parse_response_2(response)

    def read_control_params(self) -> Optional[list]:
        """
        Read control parameter command.

        This function sends a command to the motor to read the control parameter.

        Returns:
            list: Control parameter data
        """

        self._send_command(0xC0)
        return self._receive_response()

    def write_control_params(self, control_param_id: int, param_bytes: list) -> Optional[list]:
        """
        Write control parameter command.

        This function sends a command to the motor to write the control parameter.

        Args:
            control_param_id (uint8_t): Control parameter ID
            param_bytes (list): Parameter bytes
        """

        data = [0xC1, control_param_id] + param_bytes
        self._send_command(0xC1, data)
        return self._receive_response()

    def read_encoder_data(self) -> tuple:
        """
        Read encoder data command.

        This function sends a command to the motor to read the encoder data.

        Returns:
            encoder_value (uint16_t): Encoder value
            encoder_raw (uint16_t): Raw encoder value
            encoder_offset (uint16_t): Encoder offset
        """

        self._send_command(0x90)
        response = self._receive_response()
        if response:
            self.encoder_value = self._byte_to_decimal(response[2:4])
            self.encoder_raw = self._byte_to_decimal(response[4:6])
            self.encoder_offset = self._byte_to_decimal(response[6:8])
        return self.encoder_value, self.encoder_raw, self.encoder_offset

    def set_zero_position(self) -> int:
        """
        Set the current position as the zero position.

        This function sends a command to the motor to set the current position as the zero position.
        Note: the encoder offset will be updated until the motor is restarted.
        Warning: this command will effect the lifetime of the driver, do not use it frequently.

        Returns:
            encoder_offset (int16_t): Encoder offset
        """
        self._send_command(0x19)
        response = self._receive_response()
        if response:
            self.encoder_offset = self._byte_to_decimal(response[6:8])
        return self.encoder_offset

    def read_multi_turn_angle(self) -> int:
        """
        Read the multi-turn angle.

        This function sends a command to the motor to read the multi-turn angle.

        Returns:
            multi_turn_angle (int64_t): Multi-turn angle, unit: 0.01 degree/LSB
        """

        self._send_command(0x92)
        response = self._receive_response()
        if response:
            self.multi_turn_angle = self._byte_to_decimal(response[1:8])
        return self.multi_turn_angle

    def read_single_turn_angle(self) -> int:
        """
        Read the single-turn angle.

        This function sends a command to the motor to read the single-turn angle.

        Returns:
            single_turn_angle (uint32_t): Single-turn angle, range from 0 to 35999
        """

        self._send_command(0x94)
        response = self._receive_response()
        if response:
            self.single_turn_angle = self._byte_to_decimal(response[4:8])
        return self.single_turn_angle

    def set_position_to_angle(self, multi_turn_angle: int) -> Optional[list]:
        """
        Set the current position as a multi-turn angle.

        Args:
            motor_angle (int): Target angle, unit: 0.01 degree/LSB
        """

        data = [0x00] * 3 + self._decimal_to_byte(multi_turn_angle, 4)
        self._send_command(0x95, data)
        return self._receive_response()


def prime_watchdog(motors, rate_hz: float = 50.0,
                   timeout_s: Optional[float] = None, verbose: bool = True) -> bool:
    """
    DEPRECATED -- superseded by motorbus.MotorBus.arm() (and arm_motors()).

    Kept only so existing importers still run. Its original premise below is now
    known to be too strong: t7_recover_no_powercycle.py proved the input-lost
    latch (0x80) CAN be cleared over CAN with 0x9B + 0x88, so no power cycle is
    required and priming-before-power-on is no longer necessary. New code should
    use MotorBus.arm(), which streams zero torque + runs the minimal recovery
    ladder non-blocking across all motors.

    Keep the motor MCU's comms watchdog from tripping through power-on.

    CALL THIS AT THE START OF EVERY SCRIPT (after building the motor(s), before
    motor_run / the control loop), THEN switch on motor power.

    Why it was thought needed: the watchdog is implemented in the motor MCU
    firmware. It enters protect mode ~500 ms after power-on if no command has
    arrived, and the trip LATCHES the input-signal-timeout flag (status-1 error
    bit 7, 0x80). This latch CAN be cleared over CAN with 0x9B + 0x88 (clear +
    run; see motorbus.MotorBus.recover / arm_motors), so a power cycle is NOT
    required -- but streaming zero torque from before power-on still avoids the
    trip entirely.

    This streams zero torque (motor limp, no motion) to every motor at rate_hz and
    returns once each motor replies with no input-timeout latch. Run the script
    first, power the motor(s) on when prompted; the zero stream is alive from the
    first frame the MCU sees, so the watchdog never trips. If a motor was switched
    on too early and is already latched, just power-CYCLE it now -- the live stream
    prevents a re-trip and this returns as soon as it comes up clean.

    Sends are fire-and-forget with a short reply poll, so the stream stays fast
    (well under the ~500 ms window) even while the motor is still off.

    Args:
        motors: a single LKMotor, or an iterable of LKMotor sharing the bus.
        rate_hz: zero-command stream rate (50 Hz feeds a 500 ms watchdog ~10x over).
        timeout_s: give up after this many seconds (None = wait indefinitely).
        verbose: print a throttled "power on the motor" status line.

    Returns:
        bool: True once every motor is powered with no input-timeout latch;
              False on timeout.
    """
    motor_list = [motors] if isinstance(motors, LKMotor) else list(motors)
    dt = 1.0 / rate_hz
    recv_timeout = min(dt, 0.05)
    ready = {m.motor_id: False for m in motor_list}
    t_start = time.time()
    last_show = 0.0
    if verbose:
        print(f"[watchdog] streaming torque=0 to motor(s) {list(ready)} -- "
              "power them ON now (Ctrl-C to abort)...")

    while not all(ready.values()):
        t0 = time.time()
        for m in motor_list:
            m.send_zero_command()                       # fire-and-forget keep-alive
        for m in motor_list:
            if ready[m.motor_id]:
                continue
            # a reply means the motor is powered; then confirm no 0x80 latch
            if m._receive_response(timeout=recv_timeout) is not None:
                status = m.read_motor_status_1()
                if status is not None and not (status[4] & 0x80):
                    ready[m.motor_id] = True

        if timeout_s is not None and (time.time() - t_start) > timeout_s:
            if verbose:
                pend = [mid for mid, ok in ready.items() if not ok]
                print(f"\n[watchdog] timed out; not ready: {pend}")
            return False
        if verbose and t0 - last_show > 0.5:
            pend = [mid for mid, ok in ready.items() if not ok]
            print(f"  waiting for motor(s) {pend} -- power on (power-CYCLE if "
                  "already latched)   ", end="\r")
            last_show = t0
        time.sleep(max(0.0, dt - (time.time() - t0)))

    # The zero-command stream above is fire-and-forget, so each motor's unread
    # replies are backlogged in the RX queue. Drop them before the caller starts
    # reply-reading control, else telemetry lags by the whole backlog.
    for m in motor_list:
        m.flush_rx()
    if verbose:
        print("\n[watchdog] all motor(s) powered, no input-timeout fault -- ready.")
    return True


if __name__ == "__main__":
    motor = LKMotor(bus_interface="socketcan", bus_channel="can0", motor_id=1)

    status_1 = motor.read_motor_status_1()
    print(f"Motor status 1: {status_1}")

    