import sys
from PyQt5 import QtWidgets, QtCore, QtSerialPort
import laser_intensity_optimizer_11 as Optimike # type: ignore
from queue import Queue
from Pico_module import PS2000a
from Andor_iStar_FastReadout_Mock import AndorIStar
from Andor_iDus_FastReadout_Mock import AndorIDus
from PyQt5.QtCore import pyqtSignal
import time
import numpy as np
import scipy.optimize as opt
import myFunctions as my
from PyQt5.QtCore import QEventLoop
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class PlotWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(PlotWindow, self).__init__(parent)
        self.setWindowTitle("Fitting Process Visualization")
        self.setGeometry(100, 100, 800, 600)

        self.main_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.main_widget)

        # Create a matplotlib figure and canvas
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        
        # Layout to hold the canvas
        layout = QtWidgets.QVBoxLayout(self.main_widget)
        layout.addWidget(self.canvas)

        # Store the plot axes for further usage
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title('Data and Fit')
        self.ax.set_xlabel('Wavelength')
        self.ax.set_ylabel('Intensity')

    def plot_data(self, wavelengths, intensity_data):
        """ Plot the intensity data over wavelength """
        self.ax.clear()
        self.ax.plot(wavelengths, intensity_data, 'b-', label='Data')
        self.ax.set_title('Data and Fit')
        self.ax.set_xlabel('Wavelength')
        self.ax.set_ylabel('Intensity')
        self.ax.legend()
        self.canvas.draw()

    def plot_fit(self, wavelengths, fitted_curve):
        """ Plot the fitted curve """
        self.ax.plot(wavelengths, fitted_curve, 'r--', label='Fit')
        self.ax.legend()
        self.canvas.draw()

class FitParameters(QtWidgets.QDialog):   
    CommandSignal = pyqtSignal([list])

    def __init__(self, parameters=Optimike.DEFAULTFITPARAMETERS):
        super().__init__()
        self.parameters = parameters
        self.intensity_data = None
        self.device = ''
        self.initUI()

    def initUI(self):
        self.layout = QtWidgets.QGridLayout()
        self.setLayout(self.layout)
        self.setWindowTitle("Set Fit Parameters")

        self.param_widgets = {}
        self.result_labels = {}
        row = 0

        for param, value in self.parameters.items():
            self.createParameterRow(param, value, row)
            row += 1

        self.createButtons(row)
        self.setGeometry(300, 300, 600, 300)

    def createParameterRow(self, param, value, row):
        label = QtWidgets.QLabel(param)
        self.layout.addWidget(label, row, 0)

        min_edit = QtWidgets.QLineEdit(str(value['min']))
        self.layout.addWidget(min_edit, row, 1)

        max_edit = QtWidgets.QLineEdit(str(value['max']))
        self.layout.addWidget(max_edit, row, 2)

        self.param_widgets[param] = {'min': min_edit, 'max': max_edit}

        result_label = QtWidgets.QLabel("")
        self.layout.addWidget(result_label, row, 3)
        self.result_labels[param] = result_label

    def createButtons(self, row):
        self.OK_Button = QtWidgets.QPushButton("OK")
        self.layout.addWidget(self.OK_Button, row, 1)
        self.OK_Button.clicked.connect(self.onOK)

        self.Cancel_Button = QtWidgets.QPushButton("Cancel")
        self.layout.addWidget(self.Cancel_Button, row, 2)
        self.Cancel_Button.clicked.connect(self.Cancel)

        self.Fit_Button = QtWidgets.QPushButton("Fit")
        self.layout.addWidget(self.Fit_Button, row, 3)
        self.Fit_Button.clicked.connect(self.onFit)

    def dataCollecter(self):
        if self.device not in ['Picoscope', 'CCD', 'ICCD']:
            raise ValueError("Invalid device type")

        self.command_signal.emit(["Start" if self.device == 'Picoscope' else "GetAcquisition"])
        loop = QEventLoop()
        response_handler = {
            'Picoscope': self.PicoscopeResponseHandler,
            'CCD': self.CCDResponseHandler,
            'ICCD': self.ICCDResponseHandler
        }[self.device]

        self.oscilloscope.ResponseSignal.connect(lambda response: response_handler(response, loop))
        loop.exec_()

        if not hasattr(self, 'data') or self.data is None:
            raise AttributeError(f"Failed to retrieve data from {self.device} after waiting")
        
        self.intensity_data = np.array(self.data)

    def Gaussian(self, x, a1, mu1, sigma1, c):
        sigma1 /= 2.3
        return a1/(sigma1 * np.sqrt(2 * np.pi)) * np.exp(-((x - mu1)**2/(2 * sigma1**2))) + c

    def onFit(self):
        # Create and show the plot window for live data visualization
        if hasattr(self, 'plot_window') and self.plot_window.isVisible():
            self.plot_window.close()
        self.plot_window = PlotWindow(self)
        self.plot_window.show()
        self.plot_window.plot_data(self.wavelengths, self.intensity_data)
        self.collectParams()
        try:
            wavelengths_optimize = self.wavelengths[my.findNearest(self.wavelengths, self.parameters['mu']['min']):my.findNearest(self.wavelengths, self.parameters['mu']['max'])]
            intensity_data_optimize = self.intensity_data[my.findNearest(self.wavelengths, self.parameters['mu']['min']):my.findNearest(self.wavelengths, self.parameters['mu']['max'])]
            popt, _ = opt.curve_fit(
                self.Gaussian,
                wavelengths_optimize,
                intensity_data_optimize,
                bounds=([self.parameters['amp']['min'], self.parameters['mu']['min'], self.parameters['sig']['min'], self.parameters['offset']['min']],
                        [self.parameters['amp']['max'], self.parameters['mu']['max'], self.parameters['sig']['max'], self.parameters['offset']['max']])
            )

            # Generate the fitted curve
            fitted_curve = self.Gaussian(wavelengths_optimize, *popt)

            # Plot the fitted curve
            self.plot_window.plot_fit(wavelengths_optimize, fitted_curve)

            # Update result labels with the fit parameters
            self.updateResultLabels(popt)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))

    def PicoscopeResponseHandler(self, response, loop):
        if response[0]=="Acquisition":
            if response[1]=="Success":
                self.data = response[2]
                print("Data collected successfully")
                loop.quit()  # Exit the event loop
            else:
                print("Response data is not in the correct format")
    
    def ICCDResponseHandler(self, response, loop):
        # print(response)
        if response[0] == "Acquisition":
            if isinstance(response[1], (list, np.ndarray)):
                self.data = response[1]
                # print("Data collected successfully")
                loop.quit()  # Exit the event loop
            else:
                # print("Response data is not in the correct format")
                pass
    
    def CCDResponseHandler(self, response, loop):
        # print(response)
        if response[0] == "Acquisition":
            if isinstance(response[1], (list, np.ndarray)):
                self.data = response[1]
                # print("Data collected successfully")
                loop.quit()  # Exit the event loop
            else:
                # print("Response data is not in the correct format")
                pass

    def setOscilloscope(self, oscilloscope, command_signal):
        self.oscilloscope = oscilloscope
        self.command_signal = command_signal
    
    def updateResultLabels(self, popt):
        # Assuming popt returns [amp, mu, sig, offset]
        for i, param in enumerate(self.parameters.keys()):
            fitted_value = popt[i] if i < len(popt) else 'N/A'
            self.result_labels[param].setText(f"{fitted_value:.2f}")
    
    def collectParams(self):
        for param, widgets in self.param_widgets.items():
            min_widget = widgets['min']
            max_widget = widgets['max']
            
            try:
                # Retrieve and convert min and max values
                min_value = float(min_widget.text())
                max_value = float(max_widget.text())
                
                # Store the values in the output dictionary
                self.parameters[param] = {'min': min_value, 'max': max_value}
            except ValueError:
                QtWidgets.QMessageBox.warning(self, "Warning", f"Invalid value for {param}. Please enter valid numbers.")
                return
    
    def onOK(self):
        self.collectParams()
        if self.oscilloscope is not None: 
            self.oscilloscope.disconnect()
            print('oscilloscope disconnected')
        if hasattr(self, 'plot_window') and self.plot_window.isVisible():
            self.plot_window.close()
        # print(self.output_value)
        self.accept()

    def Cancel(self):
        self.reject()

class OptimizerParameters(QtWidgets.QDialog):
    def __init__(self, parameters={}):
        super().__init__()

        # Use get to define measurement_device only if it exists
        self.measurement_device = parameters.get('Measurement Device', 'Not Set')
        self.control_device = parameters.get('Control Device', 'Not Set')
        self.optimization_method = parameters.get('Optimization Method', 'Not Set')
        
        self.OnFlag = parameters.get('Mode', 'Off') == 'On'

        self.initUI()

    def initUI(self):
        self.layout = QtWidgets.QGridLayout()
        self.setLayout(self.layout)
        self.setWindowTitle("Set Optimizer Parameters")

        self.Measurement_Device_Button = QtWidgets.QPushButton("Measurement Device")
        self.layout.addWidget(self.Measurement_Device_Button, 0, 0, 1, 1)
        self.Measurement_Device_Button.clicked.connect(self.selectMeasurementDevice)

        self.Control_Device_Button = QtWidgets.QPushButton("Control Device")
        self.layout.addWidget(self.Control_Device_Button, 0, 1, 1, 1)
        self.Control_Device_Button.clicked.connect(self.selectControlDevice)
        
        self.Optimization_Method_Button = QtWidgets.QPushButton("Optimization Method")
        self.layout.addWidget(self.Optimization_Method_Button, 0, 2, 1, 1)
        self.Optimization_Method_Button.clicked.connect(self.selectOptimizationMethod)

        # Update labels based on current parameter values
        self.measurement_device_label = QtWidgets.QLabel(f"Measurement Device: {self.measurement_device}")
        self.layout.addWidget(self.measurement_device_label, 1, 0, 1, 1)
        
        self.control_device_label = QtWidgets.QLabel(f"Control Device: {self.control_device}")
        self.layout.addWidget(self.control_device_label, 1, 1, 1, 1)
        
        self.optimization_method_label = QtWidgets.QLabel(f"Optimization Method: {self.optimization_method}")
        self.layout.addWidget(self.optimization_method_label, 1, 2, 1, 1)

        self.On_Button = QtWidgets.QRadioButton("On")
        self.layout.addWidget(self.On_Button, 2, 0, 1, 1)
        self.On_Button.setChecked(self.OnFlag)

        self.Off_Button = QtWidgets.QRadioButton("Off")
        self.layout.addWidget(self.Off_Button, 2, 1, 1, 1)
        self.Off_Button.setChecked(not self.OnFlag)

        self.OK_Button = QtWidgets.QPushButton("OK")
        self.layout.addWidget(self.OK_Button, 3, 0, 1, 1)
        self.OK_Button.clicked.connect(self.onOK)

        self.Cancel_Button = QtWidgets.QPushButton("Cancel")
        self.layout.addWidget(self.Cancel_Button, 3, 1, 1, 1)
        self.Cancel_Button.clicked.connect(self.Cancel)

        self.setGeometry(300, 300, 600, 200)

    def selectMeasurementDevice(self):
        items = Optimike.DEVICELIST
        default = self.measurement_device if self.measurement_device in items else items[0]
        item, ok = QtWidgets.QInputDialog.getItem(self, "Select Measurement Device", "Measurement Device:", items, items.index(default), False)
        if ok and item:
            self.measurement_device = item
            self.measurement_device_label.setText(f"Measurement Device: {item}")

    def selectControlDevice(self):
        com_ports = [port.portName() for port in QtSerialPort.QSerialPortInfo.availablePorts()]
        if not com_ports:
            QtWidgets.QMessageBox.warning(self, "Warning", "No COM ports available.")
            return
        
        default = self.control_device if self.control_device in com_ports else com_ports[0]
        item, ok = QtWidgets.QInputDialog.getItem(self, "Select Control Device", "Control Device:", com_ports, com_ports.index(default), False)
        if ok and item:
            self.control_device = item
            self.control_device_label.setText(f"Control Device: {item}")

    def selectOptimizationMethod(self):
        items = Optimike.OPTIMIZEMETHODLIST
        default = self.optimization_method if self.optimization_method in items else items[0]
        item, ok = QtWidgets.QInputDialog.getItem(self, "Select Optimization Method", "Optimization Method:", items, items.index(default), False)
        if ok and item:
            self.optimization_method = item
            self.optimization_method_label.setText(f"Optimization Method: {item}")

    def onOK(self):
        self.selected_mode = 'On' if self.On_Button.isChecked() else 'Off'
        parameters = {
            'Mode': self.selected_mode,
            'Measurement Device': self.measurement_device,
            'Control Device': self.control_device,
            'Optimization Method': self.optimization_method
        }

        # Check if all parameters are set
        if 'Not Set' in parameters:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please set all the parameters.")
            return
        
        # Open the method-specific parameters dialog
        if self.optimization_method:
            self.method_params_dialog = OptimizationMethodParameters(self.optimization_method)
            if self.method_params_dialog.exec_() == QtWidgets.QDialog.Accepted:
                parameters['Method Parameters'] = self.method_params_dialog.parameters
                self.accept()
            else:
                self.reject()
        else:
            self.accept()

    def Cancel(self):
        self.reject()
        
class OptimizationMethodParameters(QtWidgets.QDialog):
    def __init__(self, method, parameters=Optimike.DEFAULTMETHODPARAMETERS):
        super().__init__()
        self.method = method
        self.parameters = parameters

        self.initUI()

    def initUI(self):
        self.layout = QtWidgets.QFormLayout()
        self.setLayout(self.layout)
        self.setWindowTitle(f"Set Parameters for {self.method}")

        self.param_widgets = {}
        # Set initial values
        params = self.parameters.get(self.method, {})

        for param, default_value in params.items():
            label = QtWidgets.QLabel(param)
            self.layout.addWidget(label)
            line_edit = QtWidgets.QLineEdit()
            line_edit.setText(str(self.parameters.get(param, default_value)))
            self.layout.addWidget(line_edit)
            self.param_widgets[param] = line_edit

        self.OK_Button = QtWidgets.QPushButton("OK")
        self.layout.addWidget(self.OK_Button)
        self.OK_Button.clicked.connect(self.onOK)

        self.Cancel_Button = QtWidgets.QPushButton("Cancel")
        self.layout.addWidget(self.Cancel_Button)
        self.Cancel_Button.clicked.connect(self.Cancel)

        self.setGeometry(300, 300, 400, 300)

    def onOK(self):
        self.parameters = {}
        for param, widget in self.param_widgets.items():
            try:
                self.parameters[param] = float(widget.text())
            except ValueError:
                QtWidgets.QMessageBox.warning(self, "Warning", f"Invalid value for {param}.")
                return
        self.accept()

    def Cancel(self):
        self.reject()

class GUI(QtWidgets.QMainWindow):
    Optimizer_CommandSignal = QtCore.pyqtSignal(dict)    
    CommandSignal = pyqtSignal([list])

    def __init__(self):
        super(GUI, self).__init__()
        self.optimizationParameters = {}
        self.SetupOptimization()
        self.fitObject = FitParameters()

    def SetOptimizerThread(self):
        self.Optimizer_Thread = QtCore.QThread()
        self.Optimizer_Queue = Queue()
        self.Optimizer_Object = Optimike.LaserIntensityOptimizer()
        self.Optimizer_Object.moveToThread(self.Optimizer_Thread)
        self.Optimizer_Thread.start()
        self.Optimizer_CommandSignal.connect(self.Optimizer_Object.Commandhandler)

    def SetupOptimization(self):
        self.OptimizerStarted = False
        self.OptimizerConnected = False
        self.GUI_Optimizer_Row = 0

        self.SetOptimizerThread()
        
        self.layout = QtWidgets.QGridLayout()
        self.centralWidget = QtWidgets.QWidget()
        self.centralWidget.setLayout(self.layout)
        self.setCentralWidget(self.centralWidget)

        self.Optimizer_Statuslabel = QtWidgets.QLabel("Optimizer: n/a")
        self.layout.addWidget(self.Optimizer_Statuslabel, self.GUI_Optimizer_Row, 0, 1, 1)
        self.Optimizer_Statuslabel.setStyleSheet("background-color: rgb(255,0,0);")
        self.Optimizer_Statuslabel.setAlignment(QtCore.Qt.AlignCenter)
        
        self.Optimizer_Start_Button = QtWidgets.QPushButton("Start")
        self.layout.addWidget(self.Optimizer_Start_Button, self.GUI_Optimizer_Row, 1, 1, 1)
        self.Optimizer_Start_Button.clicked.connect(self.OptimizerStart)

        self.Optimizer_Stop_Button = QtWidgets.QPushButton("Stop")
        self.layout.addWidget(self.Optimizer_Stop_Button, self.GUI_Optimizer_Row, 2, 1, 1)
        self.Optimizer_Stop_Button.clicked.connect(self.OptimizerStop)
        self.Optimizer_Stop_Button.setEnabled(False)  # Initially disabled

        self.Optimizer_Parameter_Set_Button = QtWidgets.QPushButton("Set Optimizer Parameters")
        self.layout.addWidget(self.Optimizer_Parameter_Set_Button, self.GUI_Optimizer_Row, 3, 1, 1)
        self.Optimizer_Parameter_Set_Button.clicked.connect(self.OptimizerParameterSet)
        
        self.Fit_Parameter_Set_Button = QtWidgets.QPushButton("Set Fit Parameters")
        self.layout.addWidget(self.Fit_Parameter_Set_Button, self.GUI_Optimizer_Row, 4, 1, 1)
        self.Fit_Parameter_Set_Button.clicked.connect(self.fitParameterSet)

        self.Show_Parameters_Button = QtWidgets.QPushButton("Show Optimization Parameters")
        self.layout.addWidget(self.Show_Parameters_Button, self.GUI_Optimizer_Row, 5, 1, 1)
        self.Show_Parameters_Button.clicked.connect(self.showOptimizationParameters)

    def showOptimizationParameters(self):
        # Get the optimization parameters in a structured format
        formatted_parameters = f"""
        Mode: {self.optimizationParameters.get('Mode', 'N/A')}
        Measurement Device: {self.optimizationParameters.get('Measurement Device', 'N/A')}
        Control Device: {self.optimizationParameters.get('Control Device', 'N/A')}
        Optimization Method: {self.optimizationParameters.get('Optimization Method', 'N/A')}\n
        Method Parameters:"""
        
        method = self.optimizationParameters.get('Optimization Method', 'Unknown')
        method_params = self.optimizationParameters.get('Method Parameters', {})
        
        if method == 'SGD':
            formatted_parameters += f"""
                - Max Iterations: {method_params.get('max_iterations', 'N/A')}
                - Initial Tolerance: {method_params.get('initial_tol', 'N/A')}
                - Initial Parameter: {method_params.get('initial_parameter', 'N/A')}
                - Learning Rate: {method_params.get('learning_rate', 'N/A')}
                - Beta1: {method_params.get('beta1', 'N/A')}
                - Beta2: {method_params.get('beta2', 'N/A')}
                - Epsilon: {method_params.get('epsilon', 'N/A')}
            """
        elif method == 'Nelder-Mead':
            formatted_parameters += f"""
                - Max Iterations: {method_params.get('max_iterations', 'N/A')}
                - Initial Tolerance: {method_params.get('initial_tol', 'N/A')}
                - Initial Parameter: {method_params.get('initial_parameter', 'N/A')}
                - Tolerance: {method_params.get('tol', 'N/A')}
                - Step: {method_params.get('step', 'N/A')}
            """
        elif method == 'Direction_Maximum':
            formatted_parameters += f"""
                - Max Iterations: {method_params.get('max_iterations', 'N/A')}
                - Initial Tolerance: {method_params.get('initial_tol', 'N/A')}
                - Initial Parameter: {method_params.get('initial_parameter', 'N/A')}
                - Tolerance: {method_params.get('tol', 'N/A')}
                - Step: {method_params.get('step', 'N/A')}
            """
        else:
            formatted_parameters += "\n    - No specific method parameters available for this optimization method."

    # Handling Fit Parameters
        device = self.optimizationParameters.get('Measurement Device', 'Unknown')
        fit_params = self.optimizationParameters.get('Fit Parameters', {})
        
        formatted_parameters += "\nFit Parameters:"
        
        if device == 'Picoscope':
            formatted_parameters += "- The Picoscope is used without fitting.\n"
        elif device in ['ICCD', 'CCD']:
            formatted_parameters += f"""
                - Amplitude (amp):
                    - Min: {fit_params.get('amp', {}).get('min', 'N/A')}
                    - Max: {fit_params.get('amp', {}).get('max', 'N/A')}
                - Mu:
                    - Min: {fit_params.get('mu', {}).get('min', 'N/A')}
                    - Max: {fit_params.get('mu', {}).get('max', 'N/A')}
                - Sigma (sig):
                    - Min: {fit_params.get('sig', {}).get('min', 'N/A')}
                    - Max: {fit_params.get('sig', {}).get('max', 'N/A')}
                - Offset:
                    - Min: {fit_params.get('offset', {}).get('min', 'N/A')}
                    - Max: {fit_params.get('offset', {}).get('max', 'N/A')}
            """
        else:
            formatted_parameters += "\n            - No specific fit parameters available for this optimization method.\n"



        # Display the formatted parameters in a message box or window
        msg = QtWidgets.QMessageBox()
        msg.setWindowTitle("Optimization Parameters")
        msg.setText(formatted_parameters)
        msg.exec_()

    def fitParameterSet(self):
        selected_device = self.optimizationParameters.get("Measurement Device", None)
        if selected_device not in ["ICCD", "CCD"]:
            msg_box = QtWidgets.QMessageBox()
            msg_box.setIcon(QtWidgets.QMessageBox.Warning)
            if selected_device == "Picoscope":
                msg_box.setText("You do no need a fitting for the Picoscope")
            else:
                msg_box.setText("Please set a measurement device first.")
            msg_box.addButton(QtWidgets.QMessageBox.Ok)
            msg_box.exec_()
            return

        self.Osci_transfer(self.optimizationParameters["Measurement Device"], self.fitObject)
        
        self.fitObject.plot_window = PlotWindow(self)
        self.fitObject.plot_window.show()
        self.fitObject.device = self.optimizationParameters["Measurement Device"]
        self.fitObject.dataCollecter()
        self.fitObject.wavelengths = np.arange(1, len(self.fitObject.intensity_data) + 1)
        # Plot the raw data in the new window
        self.fitObject.plot_window.plot_data(self.fitObject.wavelengths, self.fitObject.intensity_data)

        if self.fitObject.exec_() == QtWidgets.QDialog.Accepted:
            self.optimizationParameters["Fit Parameters"] = self.fitObject.parameters

        self.stopMeasurementThread()

    def OptimizerParameterSet(self):
        self.optimizerParameterSet_dialog = OptimizerParameters(parameters=self.optimizationParameters)
        if self.optimizerParameterSet_dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.optimizationParameters.update({
                'Mode': self.optimizerParameterSet_dialog.selected_mode,
                'Measurement Device': self.optimizerParameterSet_dialog.measurement_device,
                'Control Device': self.optimizerParameterSet_dialog.control_device,
                'Optimization Method': self.optimizerParameterSet_dialog.optimization_method,
                'Method Parameters': self.optimizerParameterSet_dialog.method_params_dialog.parameters if hasattr(self.optimizerParameterSet_dialog, 'method_params_dialog') else {}
            })
            self.Optimizer_Statuslabel.setText("Parameters set")
            self.Optimizer_Statuslabel.setStyleSheet("background-color: rgb(255, 255, 0);")
        else:
            self.Optimizer_Statuslabel.setText("Parameters not set")
            self.Optimizer_Statuslabel.setStyleSheet("background-color: rgb(255, 0, 0);")
        
    def OsciResponseHandler(self,response):
        if response[0]=="Acquisition":
            if response[1]=="Success":
                self.Object.data = response[2]

    def ICCDResponseHandler(self, response):
        print(response)
        if response[0] == "Acquisition":
            if isinstance(response[1], (list, np.ndarray)):  # Check if response[1] is a list or numpy array
                self.data = response[1]
                self.data_ready = True
                print("data collected")
            else:
                # print("response[1] is not an array type")
                pass
    def CCDResponseHandler(self, response):
        # print(response)
        if response[0] == "Acquisition":
            if isinstance(response[1], (list, np.ndarray)):  # Check if response[1] is a list or numpy array
                self.data = response[1]
                self.data_ready = True
                print("data collected")
            else:
                # print("response[1] is not an array type")
                pass

    def setup_measurement_device(self,device):
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
    
    def Osci_transfer(self, parameter, object):
        if parameter == 'Picoscope':
            self.Osci_Object, self.Osci_CommandSignal = self.setup_measurement_device(parameter)
        elif parameter == 'ICCD':
            self.ICCD_Object, self.ICCD_CommandSignal = self.setup_measurement_device(parameter)
        elif parameter == 'CCD':
            self.CCD_Object, self.CCD_CommandSignal = self.setup_measurement_device(parameter)
                
        # Transfer device to the optimizer
        if parameter == 'Picoscope' and hasattr(self, 'Osci_Object'):
            object.setOscilloscope(self.Osci_Object, command_signal=self.Osci_CommandSignal)
        elif parameter == 'Picoscope':
            print("PicoScope object not found")
        elif parameter == 'ICCD' and hasattr(self, 'ICCD_Object'):
            object.setOscilloscope(self.ICCD_Object, command_signal=self.ICCD_CommandSignal)
        elif parameter == 'ICCD':
            print("ICCD object not found")
        elif parameter == 'CCD' and hasattr(self, 'CCD_Object'):
            object.setOscilloscope(self.CCD_Object, command_signal=self.CCD_CommandSignal)
        elif parameter == 'CCD':
            print("CCD object not found")

    def OptimizerStart(self):
        if self.optimizationParameters == {}:
            msg_box = QtWidgets.QMessageBox()
            msg_box.setIcon(QtWidgets.QMessageBox.Warning)
            msg_box.setText("Please set parameters first.")
            set_default_button = msg_box.addButton("Set Default", QtWidgets.QMessageBox.ActionRole)
            msg_box.addButton(QtWidgets.QMessageBox.Ok)
            msg_box.exec_()
            if msg_box.clickedButton() == set_default_button:
                self.OptimizerSetDefaultParameters()
            return
        
        selected_device = self.optimizationParameters.get("Measurement Device", None)
        fit_parameters = self.optimizationParameters.get("Method Parameters", None)
        if selected_device in ["ICCD", "CCD"] and fit_parameters == None:
            msg_box = QtWidgets.QMessageBox()
            msg_box.setIcon(QtWidgets.QMessageBox.Warning)
            msg_box.setText(f"For {selected_device} you need to set fitting parameters")
            msg_box.addButton(QtWidgets.QMessageBox.Ok)
            msg_box.exec_()
            return
        
        self.Osci_transfer(self.optimizationParameters["Measurement Device"], self.Optimizer_Object)
        self.Optimizer_Start_Button.setEnabled(False)
        self.Optimizer_Stop_Button.setEnabled(True)
        self.Optimizer_Parameter_Set_Button.setEnabled(False)

        self.optimizationParameters['Action'] = 'Start'
        self.Optimizer_CommandSignal.emit(self.optimizationParameters)
        self.Optimizer_Statuslabel.setText("Optimization is executing")
        self.Optimizer_Statuslabel.setStyleSheet("background-color: rgb(0, 255, 0);")

    def stopOptimizerThread(self):
        # Check if the optimizer thread is running
        if self.Optimizer_Thread.isRunning():
            self.Optimizer_Object.stop_flag = True
            
            # Disconnect signals
            self.Optimizer_CommandSignal.disconnect()

            # Quit the thread and wait for it to finish
            self.Optimizer_Thread.quit()
            self.Optimizer_Thread.wait()

            # Ensure the thread is not running after quitting
            if self.Optimizer_Thread.isRunning():
                self.Optimizer_Thread.terminate()

    def stopMeasurementThread(self):
        # Check if the measurement thread is running
        if hasattr(self, 'MeasurementThread') and self.MeasurementThread.isRunning():
            # Disconnect signals from the measurement device
            self.CommandSignal.disconnect()

            # Quit the measurement thread and wait for it to finish
            self.MeasurementThread.quit()
            self.MeasurementThread.wait()
        

            # Ensure the thread is not running after quitting
            if self.MeasurementThread.isRunning():
                self.MeasurementThread.terminate()
            
            if hasattr(self, 'Osci_Object'):
                self.Osci_Object.__del__
            if hasattr(self, 'ICCD_Object'):
                pass    
            if hasattr(self, 'CCD_Object'):
                pass  

    def OptimizerStop(self):

        if self.Optimizer_Object.oscilloscope is not None: 
            self.Optimizer_Object.oscilloscope.disconnect()
            print('oscilloscope disconnected')

        self.stopOptimizerThread()
        self.stopMeasurementThread()

        # Reset the optimizer thread and measurement thread
        self.SetOptimizerThread()

        # Enable the start button and disable the stop and parameter set buttons
        self.Optimizer_Start_Button.setEnabled(True)
        self.Optimizer_Stop_Button.setEnabled(False)
        self.Optimizer_Parameter_Set_Button.setEnabled(True)

        # Update status label
        self.Optimizer_Statuslabel.setText("Optimization is stopped")
        self.Optimizer_Statuslabel.setStyleSheet("background-color: rgb(255, 255, 0);")

    def OptimizerSetDefaultParameters(self):
        self.optimizationParameters = {
            'Mode': 'Off',
            'Measurement Device': Optimike.DEVICELIST[1],
            'Control Device': QtSerialPort.QSerialPortInfo.availablePorts()[1].portName(),
            'Optimization Method': Optimike.OPTIMIZEMETHODLIST[0],
            'Method Parameters': Optimike.DEFAULTMETHODPARAMETERS[Optimike.OPTIMIZEMETHODLIST[0]],
            'Fit Parameters': Optimike.DEFAULTFITPARAMETERS
        }
        self.Optimizer_Statuslabel.setText("Default parameters set")
        self.Optimizer_Statuslabel.setStyleSheet("background-color: rgb(255, 255, 0);")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = GUI()
    app.aboutToQuit.connect(window.OptimizerStop)  # Ensure threads are stopped before exiting
    window.show()
    sys.exit(app.exec_())
