# newport_controller.py
import time
from qcodes_contrib_drivers.drivers.Newport.AG_UC8 import Newport_AG_UC8

class NewportController:
    """
    A class to control Newport AG_UC8 stage controllers.

    Attributes:
        name (str): Name of the Newport controller.
        com_port (str): COM port for the Newport controller.
        device (Newport_AG_UC8): The Newport AG_UC8 device instance.
        channels (list): List of channel objects from the device.
        positions (list): List of current positions for each channel's axes.
    """
    def __init__(self, name, com_port, init_angles=[[0,0],[0,0]]):
        self.name = name
        self.com_port = com_port
        self.device = Newport_AG_UC8(name, com_port)
        self.channels = [self.device.channels[0], self.device.channels[1]]
        self.positions = init_angles

    def set_amplitude(self, channel_index, axis_index, amplitude):
        channel = self.channels[channel_index]
        axis = getattr(channel, f'axis{axis_index + 1}')
        axis.step_amplitude_neg(amplitude)
        axis.step_amplitude_pos(amplitude)

    def move_relative(self, channel_index, axis_index, steps):
        """
        Move the specified axis of a channel relative to its current position.

        Args:
            channel_index (int): Index of the channel.
            axis_index (int): Index of the axis (0 or 1).
            steps (int): Number of steps to move (positive or negative).
        """

        channel = self.channels[channel_index]
        axis = getattr(channel, f'axis{axis_index + 1}')
        max_steps_per_move = 30
        total_steps = abs(steps)
        direction = 1 if steps > 0 else -1

        while total_steps > 0:
            steps_to_move = min(total_steps, max_steps_per_move)  
            time.sleep(0.05)
            axis.move_rel(steps_to_move * direction)
            total_steps -= steps_to_move

        self.positions[channel_index][axis_index]+=steps

    def get_position(self, channel_index, axis_index):
        return self.positions[channel_index][axis_index]
    
    def disconnect(self):
        self.device.close()
    