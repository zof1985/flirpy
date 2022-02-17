# lepton 3.5 purethermal camera dll imports
# the proper folder and file are defined by the __init__ file
from Lepton import CCI
from IR16Filters import IR16Capture, NewBytesFrameEvent

# python useful packages
from datetime import datetime
from typing import Tuple
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


def to_heatmap(img, colorscale=cv2.COLORMAP_JET):
    """
    convert a sampled frame to a opencv colorscaled map.

    Parameters
    ----------
    img: 2D numpy.ndarray
        the matrix containing the temperatures collected on one sample.

    colorscale: OpenCV colormap
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
    gry = cv2.applyColorMap(gry, cv2.COLORMAP_BONE)

    # converto to heatmap
    return cv2.applyColorMap(gry, cv2.COLORMAP_JET)


def custom_QLabel(
    text: str,
    alignment: str = "center",
    font_size: int = 10,
) -> None:
    """
    shortcut to QLabel creation with custom settings.

    Parameters
    ----------
    text: str
        the text of the label

    alignement: qtw.QWidget
        the object effectively visualizing the value.

    unit: str
        the unit of measurement of the value.

    font_size: int
        the font size of the labels.

    Returns
    -------
    wdg: qtw.QWidget
        a line with the label, the object and the unit of
        measurement formatted.
    """
    lbl = qtw.QLabel(text)
    lbl.setFont(qtg.QFont("Arial", font_size))
    if alignment.lower() == "center":
        lbl.setAlignment(qtc.Qt.AlignCenter)
    elif alignment.lower() == "left":
        lbl.setAlignment(qtc.Qt.AlignLeft)
    elif alignment.lower() == "right":
        lbl.setAlignment(qtc.Qt.AlignRight)
    else:
        txt = "aligment must be any between left, center, right."
        raise ValueError(txt)
    return lbl


def custom_data_pane(
    label: str,
    widget: qtw.QWidget,
    unit: str,
    font_size: int = 10,
) -> None:
    """
    generate a widget allowing the visualization of one value.

    Parameters
    ----------
    label: str
        the name of the value

    widget: qtw.QWidget
        the object effectively visualizing the value.

    unit: str
        the unit of measurement of the value.

    font_size: int
        the font size of the labels.

    Returns
    -------
    wdg: qtw.QWidget
        a line with the label, the object and the unit of
        measurement formatted.
    """
    layout = qtw.QHBoxLayout()
    title = custom_QLabel(label, "right", font_size)
    title.setFixedWidth(180)
    title.setFixedHeight(25)
    title.setAlignment(qtc.Qt.AlignVCenter)
    layout.addWidget(title)
    widget.setFixedWidth(50)
    widget.setFixedHeight(25)
    widget.setAlignment(qtc.Qt.AlignVCenter)
    layout.addWidget(widget)
    unit = custom_QLabel(unit, "left", font_size)
    unit.setFixedWidth(50)
    unit.setFixedHeight(25)
    unit.setAlignment(qtc.Qt.AlignVCenter)
    layout.addWidget(unit)
    pane = qtw.QWidget()
    pane.setLayout(layout)
    return pane


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


class PopupDialog(qtw.QDialog):
    """
    create a popup dialog to display whilst saving files.
    """

    def __init__(self, parent):
        super(PopupDialog, self).__init__(parent=parent)

        # data saving popup
        save_gif = os.path.sep.join(["_contents", "save.gif"])
        movie = qtg.QMovie(save_gif)
        animation = qtw.QLabel()
        animation.setFixedSize(256, 256)
        animation.setMovie(movie)
        movie.start()
        message = qtw.QLabel("SAVING COLLECTED DATA")
        message.setAlignment(qtc.Qt.AlignCenter)
        message.setFont(qtg.QFont("Arial", LeptonCameraWidget.font_size()))
        diag_layout = qtw.QVBoxLayout()
        diag_layout.addWidget(animation)
        diag_layout.addWidget(message)
        self.setModal(True)
        self.setLayout(diag_layout)
        self.setWindowTitle("Please wait.")
        self.hide()


class LeptonCameraWidget(qtw.QWidget):
    """
    Initialize a PySide2 widget capable of communicating to
    an pure thermal device equipped with a lepton 3.5 sensor.

    Parameters
    ----------
    sampling_frequency: float, int
        the sampling frequency in Hz for the camera readings.
        It must be <= 8.5 Hz.
    """

    # lepton camera
    _device = None

    # class variables
    _image_multiplier = 4
    _font_size = 10
    _path = ""

    # qt timer
    _timer = None
    _last_dt = None

    # widgets
    _sampling_frequency_text = None
    _camera_label = None
    _fps_label = None
    _pointer_label = None
    _quit_button = None
    _rec_button = None

    # dialogs
    _save_popup = None

    @property
    def device(self):
        """
        return the actual device.
        """
        return self._device

    @property
    def sampling_frequency(self) -> float:
        """
        return the actual sampling frequency
        """
        return self.device.sampling_frequency

    @property
    def shape(self) -> Tuple[int, int]:
        """
        return the shape of the collected images.
        """
        return self.device.shape

    @classmethod
    def font_size(cls):
        return cls._font_size

    def is_recording(self):
        """
        check if the camera is recording data.
        """
        return self.device.is_recording()

    def _start(self):
        """
        start the timer.
        """
        try:
            self._timer.stop()
            dt = int(round(1000.0 / self.device.sampling_frequency))
            self._timer.start(dt)
        except Exception:
            pass

    def set_sampling_frequency(self, sampling_frequency: float) -> None:
        """
        set the sampling frequency value and update the _dt argument.

        Parameters
        ----------
        sampling_frequency: float, int
            the new sampling frequency
        """

        # check the input
        self.device.set_sampling_frequency(sampling_frequency)
        self._sampling_frequency_text.setText(str(self.sampling_frequency))
        self._start()

    def __init__(self, device: LeptonCamera = None, parent=None) -> None:
        """
        constructor
        """
        super(LeptonCameraWidget, self).__init__(parent=parent)

        # set the lepton camera object
        if device is None:
            self._device = LeptonCamera(sampling_frequency=5)
        else:
            txt = "device must be a LeptonCamera instance."
            assert isinstance(device, LeptonCamera), txt
            self._device = device

        # camera widget
        self._camera_label = qtw.QLabel()
        self._camera_label.setMouseTracking(True)
        self._camera_label.installEventFilter(self)

        # sampling frequency pane
        self._sampling_frequency_text = qtw.QLineEdit("")
        lbl_font = qtg.QFont("Arial", self._font_size)
        self._sampling_frequency_text.setFont(lbl_font)
        self._sampling_frequency_text.setAlignment(qtc.Qt.AlignCenter)
        sampling_frequency_pane = custom_data_pane(
            label="SAMPLING FREQUENCY",
            widget=self._sampling_frequency_text,
            unit="Hz",
        )
        self._sampling_frequency_text.returnPressed.connect(
            self._update_sampling_frequency
        )
        self.set_sampling_frequency(
            sampling_frequency=self.device.sampling_frequency,
        )

        # pointer temperature
        self._pointer_label = custom_QLabel("", font_size=self._font_size)
        pointer_pane = custom_data_pane(
            "POINTER",
            self._pointer_label,
            "°C",
            font_size=self._font_size,
        )

        # camera pane
        data_layout = qtw.QHBoxLayout()
        data_layout.addWidget(sampling_frequency_pane)
        data_layout.addWidget(pointer_pane)
        camera_pane = qtw.QWidget()
        camera_pane.setLayout(data_layout)

        # button bar with both recording and exit button
        self._quit_button = qtw.QPushButton("QUIT")
        self._quit_button.clicked.connect(self._close)
        self._rec_button = qtw.QPushButton("● START RECORDING", self)
        self._rec_button.clicked.connect(self._record)
        self._rec_button.setCheckable(True)
        button_layout = qtw.QHBoxLayout()
        button_layout.addWidget(self._rec_button)
        button_layout.addWidget(self._quit_button)
        button_pane = qtw.QWidget()
        button_pane.setLayout(button_layout)

        # main layout
        layout = qtw.QVBoxLayout()
        layout.addWidget(self._camera_label)
        layout.addWidget(camera_pane)
        layout.addWidget(button_pane)
        self.setLayout(layout)
        self.setWindowTitle("LeptonCameraWidget")
        self.setWindowOpacity(1)

        # stream handlers
        self._timer = qtc.QTimer()
        self._timer.timeout.connect(self._update)

        # popup dialog
        self._save_popup = PopupDialog(self)

    def show(self):
        """
        make the widget visible.
        """
        self.device.capture(save=False)
        self._start()
        super(LeptonCameraWidget, self).show()

    def eventFilter(self, source: qtw.QWidget, event: qtc.QEvent) -> None:
        """
        calculate the temperature-related numbers.
        """
        # check if the pointer is on the image and update pointer temperature
        if source == self._camera_label:
            if event.type() == qtc.QEvent.MouseMove:
                if self.device._last is not None:
                    x, y = (event.x(), event.y())  # get the mouse coordinates

                    # rescale to the original image size
                    w_res = int(x * self.shape[1] / self._camera_label.width())
                    h_res = int(y * self.shape[0] / self._camera_label.height())

                    # update data_label with the temperature at mouse position
                    temp = self.device._last[1][h_res, w_res]
                    self._pointer_label.setText("{:0.1f}".format(temp))

            # the pointer leaves the image, thus no temperature has to be shown
            elif event.type() == qtc.QEvent.Leave:
                self._pointer_label.setText("")

        return False

    def _make_message(self, txt):
        """
        make an alert message with a given text.
        """
        msgBox = qtw.QMessageBox()
        msgBox.setIcon(qtw.QMessageBox.Warning)
        msgBox.setText(txt)
        msgBox.setFont(qtg.QFont("Arial", self._font_size))
        msgBox.setWindowTitle("ERROR")
        msgBox.setStandardButtons(qtw.QMessageBox.Ok)
        msgBox.exec()

    def _close(self) -> None:
        """
        terminate the app.
        """
        sys.exit()

    def _record(self) -> None:
        """
        start and stop the recording of the data.
        """
        if self._rec_button.isChecked():
            self.device.interrupt()
            self.device.capture(save=True)
            self._rec_button.setText("■ STOP RECORDING")
        else:
            self.device.interrupt()
            self._rec_button.setText("● START RECORDING")
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
                        self._save_popup.show()
                        self.device.save(path)
                        self._path = ".".join(path.split(".")[:-1])
                    except TypeError as err:
                        self._make_message(err)
                    finally:
                        self._save_popup.hide()

                # reset the camera buffer and restart the data streaming
                self.device.clear()
                self.device.capture(save=False)

    def _update(self) -> None:
        """
        display the last captured images.
        """
        if self.device._last is not None:

            # get the time and image
            dt, img = self.device._last

            # update the last dt if none
            if self._last_dt is None:
                self._last_dt = dt

            # convert the image to an heatmap
            heatmap = to_heatmap(img, cv2.COLORMAP_JET)

            # resize preserving the aspect ratio
            height = int(self._image_multiplier * img.shape[0])
            width = int(self._image_multiplier * img.shape[1])
            img_resized = cv2.resize(heatmap, (width, height))

            # set the recording overlay if required
            if self.is_recording():
                tt = dt - list(self.device._data.keys())[0]
                tt = tt.total_seconds()
                h, remainder = divmod(tt, 3600)
                m, remainder = divmod(remainder, 60)
                s, f = divmod(remainder, 1)
                h = int(h)
                m = int(m)
                s = int(s)
                f = int(f * 1000)
                lbl = "{:02d}:{:02d}:{:02d}.{:03d}".format(h, m, s, f)
                cv2.putText(
                    img_resized,
                    "REC: {}".format(lbl),
                    (10, int(height * 0.95)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    2,
                )

            # update fps
            den = (dt - self._last_dt).total_seconds()
            fps = 0.0 if den == 0.0 else (1.0 / den)
            cv2.putText(
                img_resized,
                "FPS: {:0.2f}".format(fps),
                (int(width * 0.7), int(height * 0.05)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                2,
            )
            self._last_dt = dt

            # update the view
            qimage = qimage2ndarray.array2qimage(img_resized)
            self._camera_label.setPixmap(qtg.QPixmap.fromImage(qimage))

    def _update_sampling_frequency(self):
        """
        update the sampling frequency according to the input value.
        """
        try:
            fs = float(self._sampling_frequency_text.text())
        except Exception:
            txt = "The inputed sampling frequency is not valid."
            self._make_message(txt)
            self.set_sampling_frequency(self.sampling_frequency)

        if fs <= 0 or fs > 8.5:
            txt = "Sampling frequency must be in the (0, 8.5] range."
            self._make_message(txt)
            self.set_sampling_frequency(self.sampling_frequency)
        else:
            self.set_sampling_frequency(fs)
