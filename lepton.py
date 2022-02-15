# lepton 3.5 purethermal camera dll imports
# the proper folder and file are defined by the __init__ file
from Lepton import CCI
from IR16Filters import IR16Capture, NewBytesFrameEvent

# python useful packages
from threading import Thread
from datetime import datetime
from typing import Tuple
import PySide2.QtWidgets as qtw
import PySide2.QtCore as qtc
import PySide2.QtGui as qtg
import qimage2ndarray
import numpy as np
import h5py
import json
import sys
import cv2
import os


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
    _recording = False
    _last = None
    _dt = 200
    _last_dt = None
    _sampling_frequency = 5
    _time_format = "%H:%M:%S.%f"
    _date_format = "%Y-%b-%d " + _time_format
    _path = ""

    def __init__(
        self,
        sampling_frequency: float = 5,
        gain_mode: str = "HIGH",
    ) -> None:
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

    def _add_frame(
        self,
        array: bytearray,
        width: int,
        height: int,
    ) -> None:
        """
        add a new frame to the buffer of readed data.
        """

        # get the sampling timestamp
        dt = datetime.now()

        # check if enough time has passes since the last sample
        if self._last_dt is None:
            delta_msec = 0
            timedelta = dt - dt
        else:
            timedelta = dt - self._last_dt
            delta_msec = int(round(timedelta.total_seconds() * 1000))
        if delta_msec >= self._dt:

            # update the last_dt
            self._last_dt = dt

            # get the timestamps
            timestamp = dt.strftime(self._date_format)
            lapsed = timedelta.strftime(self._time_format)

            # parse the thermal data to become a readable numpy array
            img = np.fromiter(array, dtype="uint16").reshape(height, width)
            img = (img - 27315.0) / 100.0  # centikelvin --> celsius conversion
            img = img.astype(np.float16)

            # get the fps
            fps = timedelta.total_seconds() ** (-1)

            # update the last reading
            labels = ["timestamp", "image", "fps", "recording_time"]
            values = [timestamp, img, fps, lapsed]
            self._last = dict(zip(labels, values))

            # update the list of collected data
            if self.is_recording():
                self._data[timestamp] = img

    def _get_last(self) -> dict:
        """
        return the last sampled data.
        """
        return self._last

    def capture_start(
        self,
        save: bool = True,
        n_frames: Tuple[int, None] = None,
        time: Tuple[int, None] = None,
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

        time: None / int
            if a positive int is provided, data is sampled for the indicated
            amount of seconds.
        """

        # start reading data
        assert save or not save, "save must be a bool"
        self._reader.RunGraph()
        while self._get_last() is None:
            pass
        self._recording = save

        # continue reading until a stopping criterion is met
        if time is not None:

            def stop_reading(time):
                if len(self._data) > 0:
                    keys = [i for i in self._data]
                    t0 = keys[0]
                    t1 = keys[-1]
                    if (t1 - t0).total_seconds() >= time:
                        self.stop()

            t = Thread(target=stop_reading, args=[time])
            t.run()

        elif n_frames is not None:

            def stop_reading(n_frames):
                if len(self._data) >= n_frames:
                    self.stop()

            t = Thread(target=stop_reading, args=[n_frames])
            t.run()

    def capture_stop(self) -> None:
        """
        stop reading from camera.
        """
        self._recording = False
        self._reader.StopGraph()

    @staticmethod
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

        # datetime converted
        def to_datetime(txt):
            return datetime.strptime(txt, "%d-%b-%Y %H:%M:%S.%f")

        # obtain the readed objects
        if extension == "json":  # json format
            with open(filename, "r") as buf:
                obj = json.load(buf)
            timestamps = list(obj.keys())
            samples = np.array(list(obj.values())).astype(np.float16)

        elif extension == "npz":  # npz format
            with np.load(filename) as obj:
                timestamps = obj["timestamps"]
                samples = obj["samples"]

        elif extension == "h5":  # h5 format
            with h5py.File(filename, "r") as obj:
                timestamps = obj["timestamps"][:].astype(str)
                samples = obj["samples"][:].astype(np.float16)

        # return the readings as dict
        return dict(zip(map(to_datetime, timestamps), samples))

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
        self._dt = int(round(1000.0 / sampling_frequency))

    def is_recording(self) -> bool:
        return self._recording

    def clear_memory(self) -> None:
        """
        clear the current object memory and buffer
        """
        self._data = {}
        self._last = None

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
            with open(filename, "w") as buf:
                json.dump(self.to_dict(), buf)

        elif extension == "npz":  # npz format
            timestamps, samples = self.to_numpy()
            np.savez(filename, timestamps=timestamps, samples=samples)

        elif extension == "h5":  # h5 format
            hf = h5py.File(filename, "w")
            times, samples = self.to_numpy()
            times = times.tolist()
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
        It must be <= 8.5 Hz.
    """

    # lepton camera
    _camera = None

    # class variables
    _image_multiplier = 2
    _font_size = 12

    # qt timer
    _timer = None

    # widgets
    _sampling_frequency_text = None
    _camera_label = None
    _fps_label = None
    _pointer_label = None
    _quit_button = None
    _rec_button = None

    # dialogs
    _save_poput = None

    @staticmethod
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
        return LeptonCamera.read_file(filename=filename)

    @property
    def sampling_frequency(self) -> float:
        """
        return the actual sampling frequency
        """
        return float(self._camera._sampling_frequency)

    @property
    def shape(self) -> Tuple[int, int]:
        """
        return the shape of the collected images.
        """
        return self._camera.shape

    def set_sampling_frequency(self, sampling_frequency: float) -> None:
        """
        set the sampling frequency value and update the _dt argument.

        Parameters
        ----------
        sampling_frequency: float, int
            the new sampling frequency
        """

        # check the input
        self._camera.set_sampling_frequency(sampling_frequency)
        self._sampling_frequency_text.insert(str(self._sampling_frequency))

    def __init__(self, sampling_frequency: float = 5) -> None:
        """
        constructor
        """
        super(LeptonCameraWidget, self).__init__()

        # set the lepton camera object
        self._camera = LeptonCamera(sampling_frequency=sampling_frequency)
        self._camera.capture_start(save=False)

        # camera widget
        self._camera_label = qtw.QLabel()
        self._camera_label.setMouseTracking(True)
        self._camera_label.installEventFilter(self)

        # sampling frequency pane
        self._sampling_frequency_text = self.QLineEdit("")
        lbl_font = qtg.QFont("Arial", self._font_size)
        self._sampling_frequency_text.setFont(lbl_font)
        self._sampling_frequency_text.setAlignment(qtc.Qt.AlignCenter)
        self._sampling_frequency_text.setFixedWidth(50)
        sampling_frequency_pane = self._data_pane(
            label="SAMPLING FREQUENCY",
            widget=self._sampling_frequency_text,
            unit="Hz",
        )
        self._sampling_frequency_text.textChanged.connect(
            self._update_sampling_frequency
        )
        self.set_sampling_frequency(sampling_frequency)

        # fps pane
        self._fps_label = self._QLabel("")
        self._fps_label.setFixedWidth(50)
        fps_pane = self._data_pane("", self._fps_label, "FPS")

        # pointer temperature
        self._pointer_label = self._QLabel("")
        self._pointer_label.setFixedWidth(50)
        pointer_pane = self._data_pane("POINTER", self._pointer_label, "°C")

        # camera pane
        data_layout = qtw.QHBoxLayout()
        data_layout.addWidget(sampling_frequency_pane)
        data_layout.addWidget(fps_pane)
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
        main_layout = qtw.QVBoxLayout()
        main_layout.addWidget(self._camera_label)
        main_layout.addWidget(camera_pane)
        main_layout.addWidget(button_pane)
        self.setLayout(main_layout)
        self.setWindowTitle("ThermoMetWidget")
        self.setWindowOpacity(1)

        # stream handlers
        self._timer = qtc.QTimer()
        self._timer.timeout.connect(self._update_image)

        # data saving popup
        save_gif = os.path.sep.join(["_contents", "save.gif"])
        movie = qtg.QMovie(save_gif)
        animation = qtw.QLabel()
        animation.setFixedSize(256, 256)
        animation.setMovie(movie)
        movie.start()
        message = qtw.QLabel("SAVING COLLECTED DATA")
        message.setAlignment(qtc.Qt.AlignCenter)
        message.setFont(qtg.QFont("Arial", self._font_size))
        diag_layout = qtw.QVBoxLayout()
        diag_layout.addWidget(animation)
        diag_layout.addWidget(message)
        diag = qtw.QDialog(self)
        diag.setModal(True)
        diag.setLayout(main_layout)
        diag.setWindowTitle("Please wait.")
        diag.hide()
        self._save_popup = diag

    def _QLabel(
        self,
        text: str,
        alignment: str = "center",
    ) -> None:
        """
        shortcut to QLabel creation with custom settings
        """
        lbl = qtw.QLabel(text)
        lbl.setFont(qtg.QFont("Arial", self._font_size))
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

    def _data_pane(
        self,
        label: str,
        widget: qtw.QWidget,
        unit: str,
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

        Returns
        -------
        wdg: qtw.QWidget
            a line with the label, the object and the unit of
            measurement formatted.
        """
        layout = qtw.QHBoxLayout()
        title = self._QLabel(label, "right")
        title.setFixedWidth(125)
        layout.addWidget(title)
        layout.addWidget(widget)
        unit = self._QLabel(unit, "left")
        unit.setFixedWidth(100)
        layout.addWidget(unit)
        pane = qtw.QWidget()
        pane.setLayout(layout)
        return pane

    def eventFilter(
        self,
        source: qtw.QWidget,
        event: qtc.QEvent,
    ) -> None:
        """
        calculate the temperature-related numbers.
        """
        # check if the pointer is on the image and update pointer temperature
        if event.type() == qtc.QEvent.MouseMove:
            x, y = (event.x(), event.y())  # get the mouse coordinates

            # rescale to the original image size
            w_res = int(x * self.shape[1] / self._camera_label.width())
            h_res = int(y * self.shape[0] / self._camera_label.height())

            # update data_label with the temperature at mouse position
            temp = self._camera._get_last()["image"][h_res, w_res]
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
            self._rec_button.setText("■ STOP RECORDING")
            self._camera._recording = True
        else:
            self._rec_button.setText("● START RECORDING")
            self._camera._recording = False
            if len(self._camera._data) > 0:

                # stop the reading
                self._camera.capture_stop()

                # let the user decide where to save the data
                file_filters = "H5 (*.h5)"
                file_filters += ";;NPZ (*.npz)"
                file_filters += ";;JSON (*.json)"
                options = qtw.QFileDialog.Options()
                options |= qtw.QFileDialog.DontUseNativeDialog
                path, ext = qtw.QFileDialog.getSaveFileName(
                    parent=self,
                    filter=file_filters,
                    dir=self.path,
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
                        self._camera.save(path)
                        self._camera.path = ".".join(path.split(".")[:-1])
                    except TypeError as err:
                        self._make_message(err)
                    finally:
                        self._save_popup.hide()

                # reset the camera buffer and restart the data streaming
                self._camera.clear()
                self._camera.capture_start(save=False)

    def _update_image(self) -> None:
        """
        display the last captured images.
        """
        obj = self._camera._get_last()
        if obj is not None:

            # get the image
            img = obj["image"]

            # convert to bone scale (flip the values)
            gry = (1 - (img - np.min(img)) / (np.max(img) - np.min(img))) * 255
            gry = np.expand_dims(gry, 2).astype(np.uint8)
            gry = cv2.merge([gry, gry, gry])
            gry = cv2.applyColorMap(gry, cv2.COLORMAP_BONE)

            # converto to heatmap
            heatmap = cv2.applyColorMap(gry, cv2.COLORMAP_JET)

            # resize preserving the aspect ratio
            h = int(self._image_multiplier * img.shape[0])
            w = int(self._image_multiplier * img.shape[1])
            img_resized = cv2.resize(heatmap, (w, h))

            # set the recording overlay if required
            if self.is_recording():
                cv2.putText(
                    img_resized,
                    "REC: {}".format(self._get_last()["recording_time"]),
                    (10, int(h * 0.95)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    2,
                )

            # update the view
            qimage = qimage2ndarray.array2qimage(img_resized)
            self._camera_label.setPixmap(qtg.QPixmap.fromImage(qimage))

            # update fps
            fps_txt = "{:0.1f}".format(self._get_last()["fps"])
            self._fps_label.setText(fps_txt)

    def _update_sampling_frequency(self):
        """
        update the sampling frequency according to the input value.
        """
        try:
            fs = float(self._sampling_frequency_text.text())
        except Exception:
            txt = "The inputed sampling frequency is not valid."
            self._make_message(txt)
            self.set_sampling_frequency(self._sampling_frequency)

        if fs <= 0 or fs > 8.5:
            txt = "Sampling frequency must be in the (0, 8.5] range."
            self._make_message(txt)
            self.set_sampling_frequency(self._sampling_frequency)
        else:
            self.set_sampling_frequency(fs)
