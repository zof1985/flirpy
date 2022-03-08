# lepton 3.5 purethermal camera dll imports
# the proper folder and file are defined by the __init__ file
from Lepton import CCI
from IR16Filters import IR16Capture, NewBytesFrameEvent

# python useful packages
from datetime import datetime
from typing import Tuple
from scipy import ndimage
import PySide2.QtWidgets as qtw
import PySide2.QtCore as qtc
import PySide2.QtGui as qtg
import qimage2ndarray
import numpy as np
import threading
import time
import h5py
import json
import sys
import cv2
import os


def read_file(filename: str) -> None:
    """
    read the recorded data from file.

    Parameters
    ----------
    filename: a valid filename path

    Returns
    -------
    obj: dict
        a dict where each key is a timestamp which contains the
        corresponding image frame

    Notes
    -----
    the file extension is used to desume which file format is used.
    Available formats are:
        - ".h5" (gzip format with compression 9)
        - ".npz" (compressed numpy format)
        - ".json"
    """

    # check filename and retrieve the file extension
    assert isinstance(filename, str), "'filename' must be a str object."
    extension = filename.split(".")[-1].lower()

    # check the extension
    valid_extensions = np.array(["npz", "json", "h5"])
    txt = "file extension must be any of " + str(valid_extensions)
    assert extension in valid_extensions, txt

    # check if the file exists
    assert os.path.exists(filename), "{} does not exists.".format(filename)

    # timestamps parsing method
    def to_datetime(txt):
        return datetime.strptime(txt, LeptonCamera.date_format())

    # obtain the readed objects
    if extension == "json":  # json format
        with open(filename, "r") as buf:
            obj = json.load(buf)
        timestamps = map(to_datetime, list(obj.keys()))
        samples = np.array(list(obj.values())).astype(np.float16)

    elif extension == "npz":  # npz format
        with np.load(filename, allow_pickle=True) as obj:
            timestamps = obj["timestamps"]
            samples = obj["samples"]

    elif extension == "h5":  # h5 format
        with h5py.File(filename, "r") as obj:
            timestamps = list(obj["timestamps"][:].astype(str))
            timestamps = map(to_datetime, timestamps)
            samples = obj["samples"][:].astype(np.float16)

    return dict(zip(timestamps, samples))


def to_heatmap(img, colormap=cv2.COLORMAP_JET):
    """
    convert a sampled frame to a opencv colorscaled map.

    Parameters
    ----------
    img: 2D numpy.ndarray
        the matrix containing the temperatures collected on one sample.

    colormap: OpenCV colormap
        the colormap to be used.

    Returns
    -------
    heatmap: 2D numpy.ndarray
        the matrix containing an heatmap representation of the provided
        sample.
    """
    # convert to bone scale (flip the values)
    gry = (1 - (img - np.min(img)) / (np.max(img) - np.min(img))) * 255
    gry = np.expand_dims(gry, 2).astype(np.uint8)
    gry = cv2.merge([gry, gry, gry])
    # gry = cv2.applyColorMap(gry, cv2.COLORMAP_BONE)

    # converto to heatmap
    return cv2.applyColorMap(gry, colormap)


class LeptonCamera:
    """
    Initialize a Lepton camera object capable of communicating to
    an pure thermal device equipped with a lepton 3.5 sensor.

    Parameters
    ----------
    sampling_frequency: float, int
        the sampling frequency in Hz for the camera readings.
        It must be <= 8.5 Hz.
    """

    # class variables
    _device = None
    _reader = None
    _data = {}
    _first = None
    _last = None
    _dt = 200
    _sampling_frequency = 5
    _time_format = "%H:%M:%S.%f"
    _date_format = "%Y-%b-%d " + _time_format

    @classmethod
    def date_format(cls):
        return cls._date_format

    @classmethod
    def time_format(cls):
        return cls._time_format

    def __init__(self, sampling_frequency: float = 5) -> None:
        """
        constructor
        """
        super(LeptonCamera, self).__init__()

        # find a valid device
        devices = []
        for i in CCI.GetDevices():
            if i.Name.startswith("PureThermal"):
                devices += [i]

        # if multiple devices are found,
        # allow the user to select the preferred one
        if len(devices) > 1:
            print("Multiple Pure Thermal devices have been found.\n")
            for i, d in enumerate(devices):
                print("{}. {}".format(i, d))
            while True:
                idx = input("Select the index of the required device.")
                if isinstance(idx, int) and idx in range(len(devices)):
                    self._device = devices[idx]
                    break
                else:
                    print("Unrecognized input value.\n")

        # if just one device is found, select it
        elif len(devices) == 1:
            self._device = devices[0]

        # tell the user that no valid devices have been found.
        else:
            self._device = None

        # open the found device
        txt = "No devices called 'PureThermal' have been found."
        assert self._device is not None, txt
        self._device = self._device.Open()
        self._device.sys.RunFFCNormalization()

        # set the gain mode
        self._device.sys.SetGainMode(CCI.Sys.GainMode.HIGH)

        # set radiometric
        try:
            self._device.rad.SetTLinearEnableStateChecked(True)
        except:
            print("this lepton does not support tlinear")

        # setup the buffer
        self._reader = IR16Capture()
        callback = NewBytesFrameEvent(self._add_frame)
        self._reader.SetupGraphWithBytesCallback(callback)

        # path init
        self._path = os.getcwd()

        # set the sampling frequency
        self.set_sampling_frequency(sampling_frequency)

    def _add_frame(self, array: bytearray, width: int, height: int) -> None:
        """
        add a new frame to the buffer of readed data.
        """
        # time data
        dt = datetime.now()

        # parse the thermal data to become a readable numpy array
        img = np.fromiter(array, dtype="uint16").reshape(height, width)
        img = (img - 27315.0) / 100.0  # centikelvin --> celsius conversion
        img = img.astype(np.float16)

        # update the last reading
        self._last = [dt, img]

    def capture(
        self,
        save: bool = True,
        n_frames: int = None,
        seconds: Tuple[float, int] = None,
    ) -> None:
        """
        record a series of frames from the camera.

        Parameters
        ----------
        save: bool
            if true the data are stored, otherwise nothing except
            "last" is updated.

        n_frames: None / int
            if a positive int is provided, n_frames are captured.
            Otherwise, all the frames collected are saved until the
            stop command is given.

        seconds: None / int
            if a positive int is provided, data is sampled for the indicated
            amount of seconds.
        """

        # check input
        assert isinstance(save, bool), "save must be a bool."
        if seconds is not None:
            txt = "'seconds' must be a float or int."
            assert isinstance(seconds, (float, int)), txt
        if n_frames is not None:
            txt = "'n_frames' must be an int."
            assert isinstance(n_frames, int), txt

        # start reading data
        self._reader.RunGraph()
        while self._last is None:
            pass

        # store the last data according to the given sampling
        # frequency
        if save:
            self._first = self._last

            def store():
                while self._first is not None:
                    self._data[self._last[0]] = self._last[1]
                    time.sleep(self._dt)

            t = threading.Thread(target=store)
            t.start()

        # continue reading until a stopping criterion is met
        if seconds is not None:

            def stop_reading(time):
                while self._first is None:
                    pass
                dt = 0
                while dt < seconds:
                    dt = (self._last[0] - self._first[0]).total_seconds()
                self.interrupt()

            t = threading.Thread(target=stop_reading, args=[time])
            t.run()

        elif n_frames is not None:

            def stop_reading(n_frames):
                while len(self._data) < n_frames:
                    pass
                self.interrupt()

            t = threading.Thread(target=stop_reading, args=[n_frames])
            t.run()

    def interrupt(self) -> None:
        """
        stop reading from camera.
        """
        self._reader.StopGraph()
        self._first = None

    @property
    def sampling_frequency(self) -> float:
        """
        return the actual sampling frequency
        """
        return float(self._sampling_frequency)

    @property
    def shape(self) -> Tuple[int, int]:
        """
        return the shape of the collected images.
        """
        return (120, 160)

    def set_sampling_frequency(self, sampling_frequency: float) -> None:
        """
        set the sampling frequency value and update the _dt argument.

        Parameters
        ----------
        sampling_frequency: float, int
            the new sampling frequency
        """

        # check the input
        txt = "'sampling frequency' must be a value in the (0, 8.5] range."
        assert isinstance(sampling_frequency, (int, float)), txt
        assert 0 < sampling_frequency <= 8.5, txt
        self._sampling_frequency = np.round(sampling_frequency, 1)
        self._dt = 1.0 / self.sampling_frequency

    def is_recording(self) -> bool:
        return self._first is not None

    def clear(self) -> None:
        """
        clear the current object memory and buffer
        """
        self._data = {}
        self._last = None
        self._first = None

    def to_dict(self) -> dict:
        """
        return the sampled data as dict with
        timestamps as keys and the sampled data as values.

        Returns
        -------
        d: dict
            the dict containing the sampled data.
            Timestamps are provided as keys and the sampled data as values.
        """
        timestamps = list(self._data.keys())
        samples = [i.tolist() for i in self._data.values()]
        return dict(zip(timestamps, samples))

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        return the sampled data as numpy arrays.

        Returns
        -------
        t: 1D numpy array
            a numpy array containing the datetime object

        x: 3D numpy array
            a 3D array where each the first dimension correspond to each
            sample.
        """
        t = np.array(list(self._data.keys()))
        x = np.atleast_3d(list(self._data.values()))
        return t, x

    def save(self, filename: str) -> None:
        """
        store the recorded data to file.

        Parameters
        ----------
        filename: a valid filename path

        Notes
        -----
        the file extension is used to desume which file format is required.
        Available formats are:

            - ".h5" (gzip format with compression 9)
            - ".npz" (compressed numpy format)
            - ".json"

        If an invalid file format is found a TypeError is raised.
        """

        # check filename and retrieve the file extension
        assert isinstance(filename, str), "'filename' must be a str object."
        extension = filename.split(".")[-1].lower()

        # ensure the folders exist
        if not os.path.exists(filename):
            root = os.path.sep.join(filename.split(os.path.sep)[:-1])
            os.makedirs(root, exist_ok=True)

        if extension == "json":  # json format
            times, samples = self.to_numpy()
            times = [i.strftime(self._date_format) for i in times]
            samples = samples.tolist()
            with open(filename, "w") as buf:
                json.dump(dict(zip(times, samples)), buf)

        elif extension == "npz":  # npz format
            timestamps, samples = self.to_numpy()
            np.savez(filename, timestamps=timestamps, samples=samples)

        elif extension == "h5":  # h5 format
            hf = h5py.File(filename, "w")
            times, samples = self.to_numpy()
            times = [i.strftime(self._date_format) for i in times]
            hf.create_dataset(
                "timestamps",
                data=times,
                compression="gzip",
                compression_opts=9,
            )
            hf.create_dataset(
                "samples",
                data=samples,
                compression="gzip",
                compression_opts=9,
            )
            hf.close()

        else:  # unsupported formats
            txt = "{} format not supported".format(extension)
            raise TypeError(txt)


class LeptonCameraWidget(qtw.QWidget):
    """
    Initialize a PySide2 widget capable of communicating to
    an pure thermal device equipped with a lepton 3.5 sensor.

    Parameters
    ----------
    sampling_frequency: float, int
        the sampling frequency in Hz for the camera readings.
        It must be <= 8.7 Hz.
    """

    # available colormaps
    _colormaps = {
        "AUTUMN": cv2.COLORMAP_AUTUMN,
        "BONE": cv2.COLORMAP_BONE,
        "JET": cv2.COLORMAP_JET,
        "WINTER": cv2.COLORMAP_WINTER,
        "RAINBOW": cv2.COLORMAP_RAINBOW,
        "OCEAN": cv2.COLORMAP_OCEAN,
        "SUMMER": cv2.COLORMAP_SUMMER,
        "SPRING": cv2.COLORMAP_SPRING,
        "COOL": cv2.COLORMAP_COOL,
        "HSV": cv2.COLORMAP_HSV,
        "PINK": cv2.COLORMAP_PINK,
        "HOT": cv2.COLORMAP_HOT,
        "PARULA": cv2.COLORMAP_PARULA,
        "MAGMA": cv2.COLORMAP_MAGMA,
        "INFERNO": cv2.COLORMAP_INFERNO,
        "PLASMA": cv2.COLORMAP_PLASMA,
        "VIRIDIS": cv2.COLORMAP_VIRIDIS,
        "CIVIDIS": cv2.COLORMAP_CIVIDIS,
        "TWILIGHT": cv2.COLORMAP_TWILIGHT,
        "TWILIGHT_SHIFTED": cv2.COLORMAP_TWILIGHT_SHIFTED,
        "TURBO": cv2.COLORMAP_TURBO,
        "DEEPGREEN": cv2.COLORMAP_DEEPGREEN,
    }

    # private variables
    _font_size = 10
    _size = 35
    _path = ""
    _angle = 0
    _zoom = 1
    _timer = None
    _dt = None
    _view = None
    _colormap = list(_colormaps.values())[0]
    _device = None

    # widgets
    frequencyBox = None
    zoomBox = None
    cameraLabel = None
    rotationButton = None
    colorBox = None
    quitButton = None
    recButton = None
    optionsPane = None
    savePopup = None
    pointerLabel = None
    fpsLabel = None
    recordLabel = None

    def getDevice(self):
        """
        return the actual device.
        """
        return self._device

    def start(self):
        """
        start the timer.
        """
        try:
            self._timer.stop()
            self._timer.start(int(round(1000.0 / self.getFrequency())))
        except Exception:
            pass

    def show(self):
        """
        make the widget visible.
        """
        self.getDevice().capture(save=False)
        self.start()
        super(LeptonCameraWidget, self).show()

    def close(self) -> None:
        """
        terminate the app.
        """
        sys.exit()

    def isRecording(self):
        """
        check if the camera is recording data.
        """
        return self.getDevice().is_recording()

    def getFrequency(self) -> float:
        """
        return the actual sampling frequency
        """
        return self.getDevice().sampling_frequency

    def setFrequency(self) -> None:
        """
        set the sampling frequency.
        """
        # update the timer time
        self._timer.stop()
        freq = self.frequencyBox.value()
        self.getDevice().set_sampling_frequency(freq)
        self._timer.start(int(round(1000.0 / freq)))

    def setAngle(self) -> None:
        """
        set the rotation angle.
        """
        self._angle += 90

    def setZoom(self) -> None:
        """
        set the actual zoom.
        """
        self._zoom = self.zoomBox.value()

    def setColor(self, text) -> None:
        """
        set the actual colormap
        """
        self._colormap = self._colormaps[text]

    def getFrame(self):
        """
        return the last frame view.
        It returns None if no value has been sampled.
        """
        return self._dt, self._view

    def __init__(self, device: LeptonCamera = None, parent=None) -> None:
        """
        constructor
        """
        super(LeptonCameraWidget, self).__init__(parent=parent)
        self.font = qtg.QFont("Arial", self._font_size)
        self._path = os.path.sep.join(__file__.split(os.path.sep)[:-1])

        # sampling frequency
        self.frequencyBox = qtw.QDoubleSpinBox()
        self.frequencyBox.setFont(self.font)
        self.frequencyBox.setDecimals(1)
        self.frequencyBox.setMinimum(1.0)
        self.frequencyBox.setSingleStep(0.1)
        self.frequencyBox.setMaximum(8.7)
        self.frequencyBox.setValue(5.0)
        self.frequencyBox.setValue(5.0)
        self.frequencyBox.valueChanged.connect(self.setFrequency)
        frequencyLayout = qtw.QHBoxLayout()
        frequencyLayout.setContentsMargins(2, 0, 2, 2)
        frequencyLayout.addWidget(self.frequencyBox)
        frequencyPane = qtw.QGroupBox("Sampling frequency (Hz)")
        frequencyPane.setLayout(frequencyLayout)

        # zoom
        self.zoomBox = qtw.QDoubleSpinBox()
        self.zoomBox.setFont(self.font)
        self.zoomBox.setDecimals(1)
        self.zoomBox.setMinimum(1)
        self.zoomBox.setMaximum(5)
        self.zoomBox.setSingleStep(0.1)
        self.zoomBox.setValue(3)
        self.zoomBox.valueChanged.connect(self.setZoom)
        zoomLayout = qtw.QHBoxLayout()
        zoomLayout.setContentsMargins(2, 0, 2, 2)
        zoomLayout.addWidget(self.zoomBox)
        zoomPane = qtw.QGroupBox("Zoom (X times)")
        zoomPane.setLayout(zoomLayout)

        # colormaps
        self.colorBox = qtw.QComboBox()
        self.colorBox.addItems(list(self._colormaps.keys()))
        self.colorBox.setFont(self.font)
        self.colorBox.currentTextChanged.connect(self.setColor)
        colorLayout = qtw.QHBoxLayout()
        colorLayout.setContentsMargins(2, 0, 2, 2)
        colorLayout.addWidget(self.colorBox)
        colorPane = qtw.QGroupBox("Colormap")
        colorPane.setLayout(colorLayout)

        # options pane
        optLine = qtw.QWidget()
        optLayout = qtw.QHBoxLayout()
        optLayout.addWidget(frequencyPane)
        optLayout.addWidget(zoomPane)
        optLayout.setSpacing(10)
        optLayout.setContentsMargins(2, 2, 2, 0)
        optLine.setLayout(optLayout)
        optionsLayout = qtw.QVBoxLayout()
        optionsLayout.addWidget(optLine)
        optionsLayout.addWidget(colorPane)
        optionsLayout.setSpacing(10)
        optionsLayout.setContentsMargins(2, 2, 2, 2)
        self.optionsPane = qtw.QWidget()
        self.optionsPane.setLayout(optionsLayout)

        # pointer label
        self.pointerLabel = qtw.QLabel("")
        self.pointerLabel.setFont(self.font)
        self.pointerLabel.setAlignment(qtc.Qt.AlignCenter | qtc.Qt.AlignVCenter)
        pointerLayout = qtw.QHBoxLayout()
        pointerLayout.setContentsMargins(2, 0, 2, 2)
        pointerLayout.addWidget(self.pointerLabel)
        pointerPane = qtw.QGroupBox("Pointer temp. (°C)")
        pointerPane.setLayout(pointerLayout)

        # fps label
        self.fpsLabel = qtw.QLabel("")
        self.fpsLabel.setFont(self.font)
        self.fpsLabel.setAlignment(qtc.Qt.AlignCenter | qtc.Qt.AlignVCenter)
        fpsLayout = qtw.QHBoxLayout()
        fpsLayout.setContentsMargins(2, 0, 2, 2)
        fpsLayout.addWidget(self.fpsLabel)
        fpsPane = qtw.QGroupBox("FPS")
        fpsPane.setLayout(fpsLayout)

        # camera rotation
        self.rotationButton = qtw.QPushButton()
        icon = os.path.sep.join([self._path, "_contents", "rotation.png"])
        icon = qtg.QIcon(qtg.QPixmap(icon).scaled(self._size, self._size))
        self.rotationButton.setIcon(icon)
        self.rotationButton.setFlat(True)
        self.rotationButton.setFixedHeight(self._size)
        self.rotationButton.setFixedWidth(self._size)
        self.rotationButton.clicked.connect(self.setAngle)
        rotationLayout = qtw.QHBoxLayout()
        rotationLayout.setContentsMargins(2, 0, 2, 2)
        rotationLayout.addWidget(self.rotationButton)
        rotationPane = qtw.QGroupBox("Rotate 90°")
        rotationPane.setLayout(rotationLayout)

        # data pane
        dataLayout = qtw.QHBoxLayout()
        dataLayout.setContentsMargins(0, 0, 0, 0)
        dataLayout.setSpacing(10)
        dataLayout.addWidget(pointerPane)
        dataLayout.addWidget(fpsPane)
        dataLayout.addWidget(rotationPane)
        dataPane = qtw.QWidget()
        dataPane.setLayout(dataLayout)

        # record pane
        self.recordLabel = qtw.QLabel("")
        self.recordLabel.setFont(self.font)
        self.recordLabel.setAlignment(qtc.Qt.AlignCenter | qtc.Qt.AlignVCenter)
        recordLayout = qtw.QHBoxLayout()
        recordLayout.setContentsMargins(2, 0, 2, 2)
        recordLayout.addWidget(self.recordLabel)
        recordPane = qtw.QGroupBox("Recording time")
        recordPane.setLayout(recordLayout)

        # quit button
        self.quitButton = qtw.QPushButton("QUIT")
        self.quitButton.setFixedHeight(self._size)
        self.quitButton.clicked.connect(self.close)

        # rec button
        self.recButton = qtw.QPushButton("● START RECORDING")
        self.recButton.setFixedHeight(self._size)
        self.recButton.clicked.connect(self.record)
        self.recButton.setCheckable(True)

        # buttons pane
        buttonLayout = qtw.QHBoxLayout()
        buttonLayout.addWidget(self.recButton)
        buttonLayout.addWidget(self.quitButton)
        buttonLayout.setSpacing(10)
        buttonLayout.setContentsMargins(2, 2, 2, 2)
        buttonPane = qtw.QWidget()
        buttonPane.setLayout(buttonLayout)

        # set the lepton camera object
        if device is None:
            self._device = LeptonCamera(sampling_frequency=5)
        else:
            txt = "device must be a LeptonCamera instance."
            assert isinstance(device, LeptonCamera), txt
            self._device = device

        # camera widget
        self.cameraLabel = qtw.QLabel()
        self.cameraLabel.setMouseTracking(True)
        self.cameraLabel.installEventFilter(self)

        # main layout
        layoutLeft = qtw.QVBoxLayout()
        layoutLeft.addWidget(self.cameraLabel)
        layoutLeft.addWidget(dataPane)
        layoutLeft.addWidget(recordPane)
        layoutLeft.addWidget(buttonPane)
        layoutLeft.setSpacing(10)
        layoutLeft.setContentsMargins(0, 0, 0, 0)
        leftPane = qtw.QWidget()
        leftPane.setLayout(layoutLeft)
        layout = qtw.QHBoxLayout()
        layout.addWidget(leftPane)
        layout.addWidget(self.optionsPane)
        layout.setSpacing(10)
        layout.setContentsMargins(2, 2, 2, 2)
        self.setLayout(layout)
        icon = os.path.sep.join([self._path, "_contents", "main.png"])
        icon = qtg.QIcon(qtg.QPixmap(icon).scaled(self._size, self._size))
        self.setWindowIcon(icon)
        self.setWindowTitle("LeptonCameraWidget")

        # data saving popup
        save_gif = os.path.sep.join([self._path, "_contents", "save.gif"])
        movie = qtg.QMovie(save_gif)
        animation = qtw.QLabel()
        animation.setFixedSize(256, 256)
        animation.setMovie(movie)
        movie.start()
        message = qtw.QLabel("SAVING COLLECTED DATA")
        message.setAlignment(qtc.Qt.AlignCenter)
        message.setFont(self.font)
        diagLayout = qtw.QVBoxLayout()
        diagLayout.addWidget(animation)
        diagLayout.addWidget(message)
        popup = qtw.QDialog()
        popup.setWindowFlags(qtc.Qt.FramelessWindowHint)
        popup.setModal(True)
        popup.setLayout(diagLayout)
        popup.setWindowTitle("Saving data")
        popup.show()
        popup.hide()
        self.savePopup = popup

        # stream handlers
        self._timer = qtc.QTimer()
        self._timer.timeout.connect(self.updateView)

        # setup the pointer temperature
        self._pointer_temp = "°C"

        # initialize the parameters
        self.setFrequency()
        self.setZoom()
        self.setColor(list(self._colormaps.keys())[0])

    def eventFilter(self, source: qtw.QWidget, event: qtc.QEvent) -> None:
        """
        calculate the temperature-related numbers.
        """
        # check if the pointer is on the image
        # and update pointer temperature
        if source == self.cameraLabel:
            if event.type() == qtc.QEvent.MouseMove:
                if self.getDevice()._last is not None:
                    view = self.getFrame()[1]
                    try:
                        temp = view[event.y(), event.x()]
                        self.pointerLabel.setText("{:0.1f}".format(temp))
                    except Exception:
                        self.pointerLabel.setText("")

            # the pointer leaves the image,
            # thus no temperature has to be shown
            elif event.type() == qtc.QEvent.Leave:
                self.pointerLabel.setText("")

        return False

    def record(self) -> None:
        """
        start and stop the recording of the data.
        """
        if self.recButton.isChecked():
            self.getDevice().interrupt()
            self.getDevice().capture(save=True)
            self.recButton.setText("■ STOP RECORDING")

        else:
            self.getDevice().interrupt()
            self.recButton.setText("● START RECORDING")
            if len(self._device._data) > 0:

                # let the user decide where to save the data
                file_filters = "H5 (*.h5)"
                file_filters += ";;NPZ (*.npz)"
                file_filters += ";;JSON (*.json)"
                options = qtw.QFileDialog.Options()
                options |= qtw.QFileDialog.DontUseNativeDialog
                path, ext = qtw.QFileDialog.getSaveFileName(
                    parent=self,
                    filter=file_filters,
                    dir=self._path,
                    options=options,
                )

                # prepare the data
                if len(path) > 0:
                    path = path.replace("/", os.path.sep)
                    ext = ext.split(" ")[0].lower()
                    if not path.endswith(ext):
                        path += "." + ext

                    # save data
                    try:
                        self.savePopup.show()
                        self.getDevice().save(path)
                        self._path = ".".join(path.split(".")[:-1])

                    except TypeError as err:
                        msgBox = qtw.QMessageBox()
                        msgBox.setIcon(qtw.QMessageBox.Warning)
                        msgBox.setText(err)
                        msgBox.setFont(qtg.QFont("Arial", self._font_size))
                        msgBox.setWindowTitle("ERROR")
                        msgBox.setStandardButtons(qtw.QMessageBox.Ok)
                        msgBox.exec()

                    finally:
                        self.savePopup.hide()

                # reset the camera buffer and restart the data streaming
                self.getDevice().clear()
                self.getDevice().capture(save=False)

    def updateView(self) -> None:
        """
        update the last frame and display it.
        """
        # no view is available
        if self.getDevice()._last is None:
            self._view = None
            self._dt = None

        else:
            dt, img = self.getDevice()._last

            # update the last datetime if required
            if self._dt is None:
                self._dt = dt

            # convert the last frame to an heatmap and apply the required
            # rotation and zoom
            img = img.astype(float)
            img = ndimage.zoom(
                input=ndimage.rotate(input=img, angle=self._angle),
                zoom=self._zoom,
            )
            h, w = img.shape
            self._view = img
            heat = to_heatmap(img, self._colormap)

            # set the recording overlay if required
            if self.isRecording():
                tt = dt - list(self.getDevice()._data.keys())[0]
                tt = tt.total_seconds()
                h, remainder = divmod(tt, 3600)
                m, remainder = divmod(remainder, 60)
                s, f = divmod(remainder, 1)
                h = int(h)
                m = int(m)
                s = int(s)
                f = int(f * 1000)
                lbl = "{:02d}:{:02d}:{:02d}.{:03d}".format(h, m, s, f)
                self.recordLabel.setText(lbl)
            else:
                self.recordLabel.setText("")

            # update the view
            qimage = qimage2ndarray.array2qimage(heat)
            self.cameraLabel.setPixmap(qtg.QPixmap.fromImage(qimage))

            # update the fps
            den = (dt - self._dt).total_seconds()
            fps = 0.0 if den == 0.0 else (1.0 / den)
            self.fpsLabel.setText("{:0.2f}".format(fps))

            # update view and datetime
            self._dt = dt

        # adjust the size
        self.adjustSize()
