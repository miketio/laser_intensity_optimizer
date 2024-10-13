import numpy as np
from scipy.optimize import minimize_scalar, minimize
import scipy.optimize as opt
import time
from Pico_module import PS2000a
from newport_controller import NewportController
from visualization_module import RealTimePlotter, MatrixPlotter
import matplotlib.pyplot as plt
from PyQt5 import QtGui, QtCore
from PyQt5.QtCore import pyqtSignal
from Andor_iStar_FastReadout_Mock import AndorIStar
from Andor_iDus_FastReadout_Mock import AndorIDus
import sys
from PyQt5.QtWidgets import QApplication
from queue import Queue
from PyQt5.QtCore import QEventLoop
import myFunctions as my

NUMBEROFCALLS = 0
OPTIMIZEMETHODLIST = ['Nelder-Mead', 'Direction_Maximum',"SGD"]
DEFAULTMETHODPARAMETERS = {
            'SGD': {
                'max_iterations': 2,
                'initial_tol': 3e-2,
                'initial_parameter': 2e2,
                'learning_rate': 10,
                'beta1': 0.9,
                'beta2': 0.99,
                'epsilon': 1e-5
            },
            'Nelder-Mead': {
                'max_iterations': 2,
                'initial_tol': 3e-2,
                'initial_parameter': 2e2,
                'tol': 3e-2,
                'step': 2e2 / 4
            },
            'Direction_Maximum': {
                'max_iterations': 2,
                'initial_tol': 3e-2,
                'initial_parameter': 2e2,
                'tol': 3e-2,
                'step': 2e2 / 4
            }
        }
DEFAULTFITPARAMETERS = {
    'amp': {'min': 2000, 'max': 3000},
    'mu': {'min': 640, 'max': 750},
    'sig': {'min': 25, 'max': 40},
    'offset': {'min': 990, 'max': 1010}
}
DEVICELIST = ['Picoscope', "CCD", "ICCD"]
OPTIMIZEMETHOD = OPTIMIZEMETHODLIST[0]

class LaserIntensityOptimizer(QtCore.QObject):
    stop_signal = pyqtSignal()
    CommandSignal = pyqtSignal([list])

    def __init__(self):
        super().__init__()
        self.best_intensity = 0.0
        self._best_angles_backup = None
        self.angles = None
        self.bounds = None
        self.oscilloscope = None  # Initialize oscilloscope to None
        self.newport = None
        self.optimize_method = None
        self.stop_flag = False
        self.get_intensity = None
        self.stop_signal.connect(self.stopThread)
        self.optimize = None
        self.optimizeParameters = None

    def setOscilloscope(self, oscilloscope, command_signal):
        self.oscilloscope = oscilloscope
        self.command_signal = command_signal

    def setObjectParameters(self, parameters,init_angles=[[0,0],[0,0]]):
        self.angles = np.array(init_angles)
        self.bounds = self.getBounds()
        try:
            self.newport = self.setupMeasurementDevice(parameters['Control Device'])
        except Exception as e:
            self.newport = None
        self.optimize_method = parameters['Optimization Method']
        
        # Set measurement device
        measurement_device = parameters['Measurement Device']
        self.get_intensity = {
            'Picoscope': self.getAverageIntensityPicoscope,
            'ICCD': self.getPeakIntensityICCD,
            'CCD': self.getPeakIntensityCCD
        }.get(measurement_device)

        # Set optimization method
        self.optimize = {
            'Direction_Maximum': self.optimizeDirectionMaximum,
            'Nelder-Mead': self.optimizeNelderMead,
            'SGD': self.optimizeSGD
        }.get(self.optimize_method)

        self.method_parameters = parameters["Method Parameters"]
        self.fit_parameters = parameters.get('Fit Parameters', {})

    def Commandhandler(self, parameters):
        print('\n', parameters)

        if parameters['Action'] == 'Stop':
            self.stop_flag = True
            return
        
        self.setObjectParameters(parameters)
        print("optimization is starting")
        if self.newport is not None and self.oscilloscope is not None:
            if parameters['Action'] == 'Start' and not self.stop_flag:
                print("optimize")
                self.optimize()
        else:
            if parameters['Action'] == 'Start' and not self.stop_flag:
                print("test")
                self.test()

    def test(self):
        #self.find_initial_point(intensity_threshold=0.15, duration=600)
        n=0
        while not self.stop_flag:
            time.sleep(1)
            print(n)
            n+=1
            if self.stop_flag:
                self.disconnectNewportController()
                 # self.disconnectMeasurementDevice()
                break
    
    def PicoscopeResponsehandler(self, response, loop):
        if response[0] == "Acquisition" and response[1] == "Success":
            self.data = response[2]  # Collect data for Picoscope
            print("Picoscope data collected successfully")
            loop.quit()

    def genericResponsehandler(self, response, loop):
        if response[0] == "Acquisition" and isinstance(response[1], (list, np.ndarray)):
            self.data = response[1]
            print("Data collected successfully")
            loop.quit()
    
    def getAverageIntensityPicoscope(self, duration=6):
        return self._getIntensity(duration, is_picoscope=True)

    def getPeakIntensityICCD(self, duration=60, num_averages=10):
        return self._getIntensity(duration, num_averages=num_averages)

    def getPeakIntensityCCD(self, duration=60, num_averages=10):
        return self._getIntensity(duration, num_averages=num_averages)

    def _getIntensity(self, duration, num_averages=1, is_picoscope=False):
        global NUMBEROFCALLS
        NUMBEROFCALLS += 1
        print(f"\nCall number {NUMBEROFCALLS}")

        if self.oscilloscope is None:
            raise ValueError(f"{'Picoscope' if is_picoscope else 'Measurement Device'} is not set")

        average_data = None
        output_value = 0
        for _ in range(num_averages):
            self.command_signal.emit(["GetAcquisition" if not is_picoscope else "Start"])

            # Create an event loop to wait for the signal to be processed
            loop = QEventLoop()
            if is_picoscope:
                self.oscilloscope.ResponseSignal.connect(lambda response: self.PicoscopeResponsehandler(response, loop))
            else:
                self.oscilloscope.ResponseSignal.connect(lambda response: self.genericResponsehandler(response, loop))

            loop.exec_()

            if not hasattr(self, 'data') or self.data is None:
                raise AttributeError("Failed to retrieve data after waiting")

            intensity_data = np.array(self.data)

            if average_data is None:
                average_data = np.zeros_like(intensity_data)

            average_data += intensity_data
            print(f"Accumulated data after acquisition {_ + 1}")

            time.sleep(duration / num_averages)

        average_data = average_data / num_averages
        
        if is_picoscope:
            output_value = np.mean(average_data / len(average_data))
        else:
            wavelengths = np.arange(1, len(average_data) + 1)
            wavelengths_optimize = wavelengths[my.findNearest(wavelengths, self.fit_parameters['mu']['min']):my.findNearest(wavelengths, self.fit_parameters['mu']['max'])]
            intensity_data_optimize = average_data[my.findNearest(wavelengths, self.fit_parameters['mu']['min']):my.findNearest(wavelengths, self.fit_parameters['mu']['max'])]
            popt, _ = opt.curve_fit(
                self.Gaussian,
                wavelengths_optimize,
                intensity_data_optimize,
                bounds=([self.fit_parameters['amp']['min'], self.fit_parameters['mu']['min'], self.fit_parameters['sig']['min'], self.fit_parameters['offset']['min']],
                        [self.fit_parameters['amp']['max'], self.fit_parameters['mu']['max'], self.fit_parameters['sig']['max'], self.fit_parameters['offset']['max']])
            )
            output_value = popt[0]

        print(f"Final result across {num_averages} acquisitions: {output_value}")

        return output_value
    
    def Gaussian(self, x, a1, mu1, sigma1, c):
        sigma1 /= 2.3
        return a1/(sigma1 * np.sqrt(2 * np.pi)) * np.exp(-((x - mu1)**2/(2 * sigma1**2))) + c
    
    def intensityFunction(self, angles, duration=2):
        if self.stop_flag:
            return float("inf")

        angles = np.round(angles).astype(int)
        self.setMirrorAngles(angles, self.newport)
        intensity = self.get_intensity(duration=duration)
        
        if intensity > self.best_intensity:
            self.best_intensity = intensity
            self._best_angles_backup = angles

        return -intensity  # Negate for minimization

    def getCustomSimplex(self, initial_point, shift=30):
        simplex = np.array([
            initial_point,
            initial_point + [shift*np.random.choice([-1, 1]), 0, 0, 0],
            initial_point + [0, shift*np.random.choice([-1, 1]), 0, 0],
            initial_point + [0, 0, shift*np.random.choice([-1, 1]), 0],
            initial_point + [0, 0, 0, shift*np.random.choice([-1, 1])]
        ])
        return simplex

    def computeGradient(self, func, step=30):
        grad = np.zeros_like(self.angles.flatten(), dtype=float)
        func0 = self.intensityFunction(self.angles)
        for i in range(self.angles.flatten().size):
            if self.stop_flag:
                return 0
            x_eps = self.angles.flatten()
            x_eps[i] += step
            grad[i] = (func(x_eps) - func0) / step
        self.setMirrorAngles(self.angles, self.newport)
        return grad

    def optimizeAlongDirection(self, func, x0, direction, method='brent', tol=1e-2, initial_alpha=1):
        def objective(alpha):
            new_point = x0 - alpha * direction * initial_alpha
            return func(new_point)
        
        result = minimize_scalar(
            lambda alpha: objective(alpha),  
            method=method,                  
            tol=tol                
        )
        return result

    def optimizeNelderMead(self):
        max_iterations=self.method_parameters["max_iterations"] 
        initial_tol=self.method_parameters["initial_tol"] 
        initial_parameter=self.method_parameters["initial_parameter"]
        tol = initial_tol
        step = initial_parameter / 4 
        iteration = 0
        new_intensity = 0
        while iteration < max_iterations:
            best_intensity_on_prev_iteration = new_intensity

            simplex = self.getCustomSimplex(self.angles.flatten(), shift=step)
            result = minimize(
                self.intensityFunction,
                self.angles,
                method='Nelder-Mead',
                options={
                    'initial_simplex': simplex,
                    'disp': True,        
                    'maxiter': 5,      
                    'fatol': tol,       
                }
            )
            self.angles = np.round(result.x).astype(int)
            self.setMirrorAngles(self.angles, self.newport)
            new_intensity = -result.fun

            relative_change = abs(new_intensity - best_intensity_on_prev_iteration) / max(abs(best_intensity_on_prev_iteration), 1)
            print(f"Iteration {iteration + 1}: New angles {self.angles} with intensity {new_intensity:.4f} (Relative change: {relative_change:.4f})")
            print(f"real intensity at this point is {self.intensityFunction(self.angles)}")
            if result.success:
                print("Convergence reached")
                break
            
            iteration += 1    
            tol = initial_tol / (iteration + 1) * 1.5
            step = initial_parameter / 4 / (iteration + 1) / 3
        
        self.disconnectNewportController()
        # self.disconnectMeasurementDevice()
        
        return self.angles, new_intensity
    
    def optimizeDirectionMaximum(self):
        max_iterations=self.method_parameters["max_iterations"]
        initial_tol=self.method_parameters["initial_tol"]
        initial_parameter=self.method_parameters["initial_parameter"]
        tol = initial_tol
        step = initial_parameter / 4 
        
        iteration = 0
        new_intensity = 0
        while iteration < max_iterations:

            best_intensity_on_prev_iteration = new_intensity
            best_angles_on_prev_iteration = self.angles

            gradient = self.compute_gradient(self.intensityFunction, step=step)
            norm = np.linalg.norm(gradient)
            if norm == 0:
                print("Gradient is zero; optimization stopped.")
                break
            direction = gradient / norm

            result = self.optimizeAlongDirection(self.intensityFunction, best_angles_on_prev_iteration.flatten(), direction, initial_alpha=step, tol=tol)
            self.angles = best_angles_on_prev_iteration - result.x * direction.reshape((2,2)) * step
            self.angles = np.round(self.angles).astype(int)    
            new_intensity = -result.fun

            relative_change = abs(new_intensity - best_intensity_on_prev_iteration) / max(abs(best_intensity_on_prev_iteration), 1)
            print(f"Iteration {iteration + 1}: New angles {self.angles} with intensity {new_intensity:.4f} (Relative change: {relative_change:.4f})")
            print(f"real intensity at this point is {self.intensityFunction(self.angles)}")
            if relative_change < tol:
                print("Convergence reached")
                break
            
            iteration += 1    
            tol = initial_tol / (iteration + 1) * 1.5
            step = initial_parameter / 4 / (iteration + 1)
        
        self.disconnectNewportController()
        # self.disconnectMeasurementDevice()
        
        return self.angles, new_intensity
    
    def optimizeSGD(self):
        max_iterations= self.method_parameters["max_iterations"]
        initial_parameter=self.method_parameters["initial_parameter"]
        initial_tol = self.method_parameters["initial_tol"]
        tol = initial_tol
        step = initial_parameter / 4 
        new_intensity = 0

        learning_rate =self.method_parameters["learning_rate"]
        beta1 = self.method_parameters["beta1"]
        beta2 =self.method_parameters["beta2"]
        epsilon = self.method_parameters["epsilon"]
        
        new_intensity = 0
        iteration = 0
        m = np.zeros_like(self.angles.flatten())
        v = np.zeros_like(self.angles.flatten())

        while iteration < max_iterations*10:
            best_intensity_on_prev_iteration = new_intensity

            gradient = self.computeGradient(self.intensityFunction, step=step)
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * (gradient ** 2)
            m_hat = m / (1 - beta1 ** (iteration + 1))
            v_hat = v / (1 - beta2 ** (iteration + 1))
            # print(learning_rate * m_hat / (np.sqrt(v_hat) + epsilon))
            self.angles = self.angles.flatten() - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            self.angles = np.round(self.angles).astype(int).reshape((2, 2))
            self.setMirrorAngles(self.angles, self.newport)
            new_intensity = -self.intensityFunction(self.angles)

            relative_change = abs(new_intensity - best_intensity_on_prev_iteration) / max(abs(best_intensity_on_prev_iteration), 1)
            if relative_change < tol:
                print("Convergence reached")
                break
            
            iteration += 1    
            step = initial_parameter /  (iteration + 1)
        
        
        self.disconnectNewportController()
        # self.disconnectMeasurementDevice()
        
        return self.angles, new_intensity

    def set_best_amplitude(self):
        new_intensity = -self.intensityFunction(self._best_angles_backup) 
        print("Best angles reached")
        print(f"Intensity difference {new_intensity - self.best_intensity}") 
    
    def find_initial_point(self, intensity_threshold=0.15, duration=600, range=300):
        start_time = time.time()
        test_angles = np.zeros(4).reshape((2,2))
        best_angles = test_angles
        best_intensity = -float('inf')

        while time.time() - start_time < duration:

            self.setMirrorAngles(test_angles, self.newport)
            current_intensity = self.get_intensity(duration=0.5)
            
            print(f"Testing angles: {test_angles}, Intensity: {current_intensity:.2f}")
            
            if current_intensity > intensity_threshold:
                if current_intensity > best_intensity:
                    best_intensity = current_intensity
                    best_angles = test_angles
                    print(f"New best angles found: {best_angles} with intensity: {best_intensity:.2f}")
                
                self.angles = best_angles
                break
            test_angles = np.random.randint(-range, range, size=4).reshape((2,2))

        if best_intensity <= intensity_threshold:
            raise ValueError("No starting point found with the desired intensity.")
    
    def checkPiezoModule(self):
        shifts = np.arange(10, 110, 10)
        durations = np.arange(1, 10)
        axes = np.arange(0, 4)

        for axis in axes:
            deviation_matrix = np.zeros((len(durations), len(shifts)))
            for i, duration in enumerate(durations):
                initial_intensity = self.intensityFunction(self.angles, duration=duration)
                
                for j, shift in enumerate(shifts):
                    shiftMatrix = np.zeros(4)
                    shiftMatrix[axis] += shift
                    shiftMatrix = shiftMatrix.reshape((2,2))
                    self.setMirrorAngles(self.angles + shiftMatrix, self.newport)
                    time.sleep(1)
                    current_intensity = self.intensityFunction(self.angles, duration=duration)
                    deviation_matrix[i, j] = abs((current_intensity - initial_intensity) / initial_intensity * 100)
            
            # print(deviation_matrix)
            MatrixPlotter.deviationMatrixPlot(durations=durations, shifts=shifts, deviationMatrix=deviation_matrix)

    def getBounds(self, shift=400):
        bounds = np.vstack([self.angles.flatten() - shift, self.angles.flatten() + shift])
        return bounds.transpose()

    def setMirrorAngles(self, angles, newport):
        print(f"Setting mirrors to angles: {angles}")
        for channel_index, (angle1, angle2) in enumerate(angles.reshape([2,2])):
            new_position1 = angle1
            new_position2 = angle2 
            print(f"Channel {channel_index + 1}: Axis 1 should move to new position {new_position1}, Axis 2 should move to new position {new_position2}")

            position1 = newport.get_position(channel_index, 0)
            position2 = newport.get_position(channel_index, 1)
            steps1 = new_position1 - position1
            steps2 = new_position2 - position2
            print(f"Channel {channel_index + 1}: Axis 1 should make {steps1} steps, Axis 2 should make {steps2} steps")

            newport.move_relative(channel_index, 0, steps1)
            newport.move_relative(channel_index, 1, steps2)

            position1 = newport.get_position(channel_index, 0)
            position2 = newport.get_position(channel_index, 1)
            print(f"Channel {channel_index + 1}: Axis 1 moved to position {position1}, Axis 2 moved to position {position2}")

    def setupMeasurementDevice(self, com, init_angles=[[0, 0], [0, 0]]):
        com = 'ASRL' + com[3]
        print(com)
        return NewportController("Newport", com, init_angles=init_angles)
    
    def disconnectNewportController(self):
        if self.newport is not None:
            self.newport.disconnect()
            print("serial port was disconnected")
    
    def disconnectMeasurementDevice(self):
        if self.oscilloscope is not None: 
            self.oscilloscope.disconnect()
            print('oscilloscope diconnected')

    def stopThread(self):
        self.stop_flag = True



class OuterSetup(QtCore.QObject):
    CommandSignal = pyqtSignal([list])


    def __init__(self):
        super().__init__()

    def OsciResponseHandler(self,response):
        if response[0]=="Acquisition":
            if response[1]=="Success":
                self.Object.data = response[2]

    def ICCDResponseHandler(self, response):
        # print(response)
        if response[0] == "Acquisition":
            if isinstance(response[1], (list, np.ndarray)):  # Check if response[1] is a list or numpy array
                self.data = response[1]
                self.data_ready = True
                print("data collected")
            else:
                print("response[1] is not an array type")
    def CCDResponseHandler(self, response):
        # print(response)
        if response[0] == "Acquisition":
            if isinstance(response[1], (list, np.ndarray)):  # Check if response[1] is a list or numpy array
                self.data = response[1]
                self.data_ready = True
                print("data collected")
            else:
                print("response[1] is not an array type")

    def setupMeasurementDevice(self,device):
        print("running")
        self.MeasurementThread = QtCore.QThread()	
        self.MeasurementQueue = Queue()
        if device == 'Picoscope':
            self.Object = PS2000a()
            self.Object.ResponseSignal.connect(self.OsciResponseHandler)
            self.data = []
            self.data_ready = False
        elif device == 'ICCD':
            self.Object = AndorIStar(self.MeasurementQueue, 'LargeChamber')
            self.Object.ResponseSignal.connect(self.ICCDResponseHandler)
        elif device == 'CCD':
            self.Object = AndorIDus(self.MeasurementQueue, 'LargeChamber')
            self.Object.ResponseSignal.connect(self.CCDResponseHandler)

        self.Object.moveToThread(self.MeasurementThread)		
        self.CommandSignal.connect(self.Object.Commandhandler)
        self.MeasurementThread.start()
        print("running")
        # Create oscilloscope object and connect
        if device == 'Picoscope':
            self.CommandSignal.emit(["Connect"])
            response = [
                2,   # Channel A Range
                'DC',   # Channel A Coupling
                2.0,   # Channel B Range
                'DC',   # Channel B Coupling
                'A',   # Trigger Source
                0.02,   # Trigger Level
                'Rising',   # Trigger Type
                0.5,   # Trigger Delay
                1.0,   # Observation Duration (seconds)
                1000    # Number of Samples
            ]
            self.CommandSignal.emit(["Settings",response])
        elif device == 'ICCD':
            self.CommandSignal.emit(["Connect"])
            self.CommandSignal.emit(['SetAcquisitionSettings','FVB','Video',0.1,'On','4x','50 kHz',1,0.01,0,0.01])
            self.CommandSignal.emit(["SetMCPGain", 0])
            self.CommandSignal.emit(["SetCooling", "Off"])
        elif device == 'CCD':
            self.CommandSignal.emit(["Connect"])
            self.CommandSignal.emit(['SetAcquisitionSettings','FVB','Video',0.1,'1.7x','100 kHz',0.05,1,0.01,0,0.0,1.5,10])
            self.CommandSignal.emit(["SetMCPGain", 0])
            self.CommandSignal.emit(["SetCooling", "Off"])
        return self.Object, self.CommandSignal
        

def main():
    # Example usage of LaserIntensityOptimizer
    app = QApplication(sys.argv)
    parameters = {
        'Control Device': 'COM6',
        'Optimization Method': 'SGD',
        'Measurement Device': 'CCD',
        'Action': 'Start',
        'Method Parameters': {
                'max_iterations': 2,
                'initial_tol': 3e-2,
                'initial_parameter': 2e2,
                'learning_rate': 10,
                'beta1': 0.9,
                'beta2': 0.99,
                'epsilon': 1e-5
            },
        'Fit Parameters': {
                'amp': {'min': 2000, 'max': 3000},
                'mu': {'min': 640, 'max': 750},
                'sig': {'min': 25, 'max': 40},
                'offset': {'min': 990, 'max': 1010}
            }
    }
    outer_setup = OuterSetup()
    oscilloscope, command_signal = outer_setup.setupMeasurementDevice(parameters["Measurement Device"])
    
    optimizer = LaserIntensityOptimizer()
    optimizer.setOscilloscope(oscilloscope, command_signal)
    optimizer.Commandhandler(parameters)

if __name__ == "__main__":
    main()