# imports
from Lepton import CCI
from typing import Tuple
from scipy import ndimage
from datetime import datetime
from IR16Filters import IR16Capture, NewBytesFrameEvent
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

import PySide2.QtWidgets as qtw
import PySide2.QtCore as qtc
import PySide2.QtGui as qtg
import numpy as np
import matplotlib
import sys

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as plc
import threading
import h5py
import json
import time
import os

# default options for matplotlib
plt.rc("font", size=3)  # controls default text sizes
plt.rc("axes", titlesize=3)  # fontsize of the axes title
plt.rc("axes", labelsize=3)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=3)  # fontsize of the x tick labels
plt.rc("ytick", labelsize=3)  # fontsize of the y tick labels
plt.rc("legend", fontsize=3)  # legend fontsize
plt.rc("figure", titlesize=3)  # fontsize of the figure title
plt.rc("figure", autolayout=True)
font = qtg.QFont("Arial", 12)


def get_QIcon(path, size=40):
    """
    return a QIcon with the required size from file.

    Parameters
    ----------
    path: str
        the file to be used as icon.

    size: int
        the dimension of the icon.

    Returns
    -------
    icon: QIcon
        the icon.
    """
    # check the entries
    assert os.path.exists(path), "path must be a valid file."
    assert isinstance(size, int), "size must be an int."

    # get the icon
    pixmap = qtg.QPixmap(path)
    scaled = pixmap.scaledToHeight(
        size,
        mode=qtc.Qt.TransformationMode.SmoothTransformation,
    )
    return qtg.QIcon(scaled)


class LeptonCamera:
    """
    Initialize a Lepton camera object capable of communicating to
    an pure thermal device equipped with a lepton 3.5 sensor.
    """

    # class variables
    _device = None
    _reader = None
    _data = {}
    _path = ""
    _first = None
    _last = None
    _dt = 200
    _angle = 0
    _sampling_frequency = 5

    @staticmethod
    def time_format():
        return "%H:%M:%S.%f"

    @staticmethod
    def date_format():
        return "%Y-%b-%d"

    @staticmethod
    def datetime_format():
        return LeptonCamera.date_format() + " " + LeptonCamera.time_format()

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

        # timestamps parsing method
        def to_datetime(txt):
            return datetime.strptime(txt, LeptonCamera.datetime_format())

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

    def __init__(self) -> None:
        """
        constructor
        """
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
        self._path = os.path.sep.join(__file__.split(os.path.sep)[:-4])

        # set the sampling frequency
        self.set_sampling_frequency(5)

        # set the rotation angle
        self.set_angle(0)

    def _add_frame(self, array: bytearray, width: int, height: int) -> None:
        """
        add a new frame to the buffer of readed data.
        """
        dt = datetime.now()  # time data
        img = np.fromiter(array, dtype="uint16").reshape(height, width)  # parse
        img = (img - 27315.0) / 100.0  # centikelvin --> celsius conversion
        img = ndimage.rotate(img, angle=self.angle, reshape=True)  # rotation
        self._last = [dt, img.astype(np.float16)]  # update the last reading

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
            self._data[self._last[0]] = self._last[1]

            def store():
                old = list(self._data.keys())[-1]
                processing_time = 0
                n = 0
                while self._first is not None:
                    dt = (self._last[0] - old).total_seconds()
                    if dt >= self._dt - processing_time:
                        tic = time.time()
                        self._data[self._last[0]] = self._last[1]
                        old = self._last[0]
                        toc = time.time()
                        tm = toc - tic
                        processing_time = processing_time * n + tm
                        n += 1
                        processing_time /= n

            t = threading.Thread(target=store)
            t.start()

        # continue reading until a stopping criterion is met
        if seconds is not None:

            def stop_reading(time):
                while self._first is None:
                    pass
                dt = 0
                while dt < time:
                    dt = (self._last[0] - self._first[0]).total_seconds()
                self.interrupt()

            t = threading.Thread(target=stop_reading, args=[seconds])
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

    def set_angle(self, angle: float) -> None:
        """
        set the rotation angle in degrees.

        Parameters
        ----------
        angle: float
            the rotation angle in degrees.
        """
        assert isinstance(angle, (int, float)), "'angle' must be a float."
        self._angle = angle

    @property
    def angle(self) -> float:
        """
        return the rotation angle
        """
        return self._angle

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
            times = [i.strftime(LeptonCamera.datetime_format()) for i in times]
            samples = samples.tolist()
            with open(filename, "w") as buf:
                json.dump(dict(zip(times, samples)), buf)

        elif extension == "npz":  # npz format
            timestamps, samples = self.to_numpy()
            np.savez(filename, timestamps=timestamps, samples=samples)

        elif extension == "h5":  # h5 format
            hf = h5py.File(filename, "w")
            times, samples = self.to_numpy()
            times = [i.strftime(LeptonCamera.datetime_format()) for i in times]
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


class Runnable(qtc.QRunnable):
    """
    generalize a runnable object to start multithreading functions.

    Parameters
    ----------
    fun: function
        the function to be runned

    **kwargs: any
        optional parameters that are passed to fun as it starts
    """

    fun = None
    kwargs = {}

    def __init__(self, fun, **kwargs):
        super().__init__()
        self.fun = fun
        self.kwargs = kwargs

    def run(self):
        return self.fun(**self.kwargs)


class RecordingWidget(qtw.QWidget):
    """
    Initialize a PySide2 widget capable showing a checkable button for
    recording things and showing the recording time.
    """

    button = None
    label = None
    start_time = None
    timer = None
    label_format = "{:02d}:{:02d}:{:02d}"
    started = qtc.Signal()
    stopped = qtc.Signal()
    _size = 50

    def __init__(self):
        super().__init__()

        # generate the output layout
        layout = qtw.QHBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 0)

        # recording time
        self.label = qtw.QLabel(self.label_format.format(0, 0, 0, 0))
        self.label.setFont(font)

        # rec button
        path = os.path.sep.join(__file__.split(os.path.sep)[:-1])
        rec_file = os.path.sep.join([path, "_contents", "rec.png"])
        rec_icon = get_QIcon(rec_file, self._size)
        self.button = qtw.QPushButton()
        self.button.setFlat(True)
        self.button.setCheckable(True)
        self.button.setContentsMargins(0, 0, 0, 0)
        self.button.setIcon(rec_icon)
        self.button.setFixedHeight(self._size)
        self.button.setFixedWidth(self._size)
        self.button.clicked.connect(self.clicked)

        # generate the output
        layout.addWidget(self.button)
        layout.addWidget(self.label)
        self.setLayout(layout)

        # start the timer runnable
        self.timer = qtc.QTimer()
        self.timer.timeout.connect(self.update_time)

    def update_time(self):
        """
        timer function
        """
        if self.start_time is not None:
            now = time.time()
            delta = now - self.start_time
            h = int(delta // 3600)
            m = int((delta - h * 3600) // 60)
            s = int((delta - h * 3600 - m * 60) // 1)
            self.label.setText(self.label_format.format(h, m, s))
        else:
            self.label.setText(self.label_format.format(0, 0, 0))

    def clicked(self):
        """
        function handling the clicking of the recording button.
        """
        if self.button.isChecked():  # the recording starts
            self.start_time = time.time()
            self.timer.start(1000)
            self.started.emit()

        else:  # the recording stops
            self.start_time = None
            self.stopped.emit()


class HoverWidget(qtw.QWidget):
    """
    defines a hover pane to be displayed over a matplotlib figure.
    """

    # class variable
    labels = {}
    formatters = {}
    artists = {}
    layout = None

    def __init__(self):
        super().__init__()
        self.layout = qtw.QGridLayout()
        self.layout.setHorizontalSpacing(20)
        self.layout.setVerticalSpacing(10)
        self.layout.setContentsMargins(10, 10, 10, 10)
        flags = qtc.Qt.FramelessWindowHint | qtc.Qt.WindowStaysOnTopHint
        self.setWindowFlags(flags)

    def add_label(
        self,
        name: str,
        unit: str,
        digits: int,
    ):
        """
        add a new label to the hover.

        Parameters
        ----------
        name: str
            the name of the axis

        unit: str
            the unit of measurement to be displayed.

        digits: int
            the number of digits to be displayed by the hover.
        """
        # check the entries
        assert isinstance(name, str), "name must be a str."
        assert isinstance(unit, str), "unit must be a str."
        assert isinstance(digits, int), "digits must be an int."
        assert digits >= 0, "digits must be >= 0."

        # add the new label
        n = len(self.labels)
        name_label = qtw.QLabel(name)
        name_label.setAlignment(qtc.Qt.AlignVCenter | qtc.Qt.AlignRight)
        name_label.setFont(font)
        self.layout.addWidget(name_label, n, 0)
        self.labels[name] = qtw.QLabel("")
        self.labels[name].setAlignment(qtc.Qt.AlignVCenter | qtc.Qt.AlignLeft)
        self.labels[name].setFont(font)
        self.layout.addWidget(self.labels[name], n, 1)
        self.setLayout(self.layout)
        self.formatters[name] = lambda x: self.unit_formatter(x, unit, digits)

    def update(self, **labels):
        """
        update the hover parameters.

        Parameters
        ----------
        labels: any
            keyworded values to be updated
        """
        for label, value in labels.items():
            self.labels[label].setText(self.formatters[label](value))

    def unit_formatter(
        self,
        x: Tuple[int, float],
        unit: str = "",
        digits: int = 1,
    ):
        """
        return the letter linked to the order of magnitude of x.

        Parameters
        ----------
        x: int, float
            the value to be visualized

        unit: str
            the unit of measurement

        digits: int
            the number of digits required when displaying x.
        """

        # check the entries
        assert isinstance(x, (int, float)), "x must be a float or int."
        assert isinstance(unit, str), "unit must be a str."
        assert isinstance(digits, int), "digits must be an int."
        assert digits >= 0, "digits must be >= 0."

        if unit != "":

            # get the magnitude
            mag = np.floor(np.log10(abs(x))) * np.sign(x)

            # scale the magnitude
            unit_letters = ["p", "n", "μ", "m", "", "k", "M", "G", "T"]
            unit_magnitudes = np.arange(-12, 10, 3)
            index = int((min(12, max(-12, mag)) + 12) // 3)

            # get the value
            v = x / (10.0 ** unit_magnitudes[index])
            letter = unit_letters[index]
        else:
            v = x
            letter = ""

        # return the value formatted
        return ("{:0." + str(digits) + "f} {}{}").format(v, letter, unit)


class FigureWidget(FigureCanvasQTAgg):
    """
    Generate a QWidget incorporating a matplotlib Figure.
    Animated artists can be provided to ensure optimal performances.

    Parameters
    ----------
    figure: matplotlib Figure
        the figure to be rendered.

    hover_offset_x: float
        the percentage of the screen width that offsets the hover with respect
        to the position of the mouse.

    hover_offset_y: float
        the percentage of the screen height that offsets the hover with respect
        to the position of the mouse.
    """

    # class variables
    _artists = {}
    hover_widget = None
    _background = None
    _fig = None
    _cid = None
    _res = None
    _ax_in = None
    _ax_out = None
    _ax_move = None
    hover_offset_x = None
    hover_offset_y = None
    event = None

    def __init__(self, hover_offset_x=0.02, hover_offset_y=0.02):
        fig = plt.figure(dpi=300)
        super().__init__(fig)
        self._fig = fig
        self._background = None
        self._res = self.figure.canvas.mpl_connect(
            "resize_event",
            self._resize_event,
        )
        self._cid = self.figure.canvas.mpl_connect(
            "draw_event",
            self.on_draw,
        )

        # mouse tracking
        self._ax_in = self.figure.canvas.mpl_connect(
            "axes_enter_event",
            self.enter_event,
        )
        self._ax_out = self.figure.canvas.mpl_connect(
            "axes_leave_event",
            self.leave_event,
        )
        self._ax_move = self.figure.canvas.mpl_connect(
            "motion_notify_event",
            self.move_event,
        )

        # create the hover mask
        self.hover_widget = HoverWidget()
        self.hover_widget.setVisible(False)
        height = self.frameSize().height()
        width = self.frameSize().width()
        self.hover_offset_x = int(round(hover_offset_x * width))
        self.hover_offset_y = int(round(hover_offset_y * height))

    def add_artist(
        self,
        artist: matplotlib.artist.Artist,
        name: str,
    ) -> None:
        """
        add a new (animated) artist to the object.

        Parameters
        ----------
        artist: matplotlib.artist.Artist
            the artist to be added

        name: str
            the name of the axis
        """
        txt = "artist must be a matplotlib.artist.Artist."
        assert isinstance(artist, matplotlib.artist.Artist), txt
        assert isinstance(name, str), "name must be a str."
        self._artists[name] = artist

    def on_draw(self, event):
        """
        Callback to register with 'draw_event'.
        """
        if event is not None:
            if event.canvas != self.figure.canvas:
                raise RuntimeError
        bbox = self.figure.canvas.figure.bbox
        self._background = self.figure.canvas.copy_from_bbox(bbox)
        self._draw_animated()

    def _draw_animated(self):
        """
        Draw all of the animated artists.
        """
        for a in self._artists.values():
            self.figure.canvas.figure.draw_artist(a)

    def update_view(self):
        """
        Update the screen with animated artists.
        """

        # update the background if required
        if self._background is None:
            self.on_draw(None)

        else:

            # restore the background
            self.figure.canvas.restore_region(self._background)

            # draw all of the animated artists
            self._draw_animated()

            # update the GUI state
            self.figure.canvas.blit(self.figure.canvas.figure.bbox)

        # let the GUI event loop process anything it has to do
        self.figure.canvas.flush_events()

    def update_hover(
        self,
        event: matplotlib.backend_bases.Event,
        **values,
    ) -> None:
        """
        update the hover position.

        Parameters
        ----------
        event: matplotlib.backend_bases.Event
            the event causing the hover update.

        values: any
            the values to be used for updating the hover.
        """
        if event is None:
            self.leave_event()
        else:
            ymax = event.canvas._lastKey[1]
            y = ymax - event.y + self.hover_offset_y
            x = event.x + self.hover_offset_x
            pnt = self.mapToGlobal(qtc.QPoint(x, y))
            self.hover_widget.move(pnt.x(), pnt.y())
            self.hover_widget.update(**values)
            self.event = event

    def _resize_event(self, event=None):
        """
        handle object resizing.
        """
        self.figure.tight_layout()
        self.figure.canvas.draw()

    def enter_event(self, event=None):
        """
        handle the entry of the mouse in the area.
        """
        self.move_event(event)

    def leave_event(self, event=None):
        """
        handle the leaving of the mouse from the area.
        """
        self.hover_widget.setVisible(False)
        self.event = None

    def move_event(self, event=None):
        """
        handle the movement of the mouse over the area.
        """
        n = len(self.hover_widget.labels)
        if not self.hover_widget.isVisible() and n > 0:
            self.hover_widget.setVisible(True)
        self.update_hover(event)


class ThermalImageWidget(FigureWidget):
    """
    defines a thermal image object.

    Parameters
    ----------
    colormap: str
        any valid matplotlib colormap.
    """

    # class variables
    _old = time.time()
    data = np.atleast_2d([])
    event = None
    colorbar = None
    _colorbar_formatter = "{:0.1f}°C"
    bounds = [1e5, -1e5]
    _ax = None

    def __init__(
        self,
        colormap="viridis",
        hover_offset_x=0.02,
        hover_offset_y=0.02,
    ) -> None:
        super().__init__(
            hover_offset_x=hover_offset_x,
            hover_offset_y=hover_offset_y,
        )
        self._ax = self._fig.add_subplot(1, 1, 1)
        self._ax.set_axis_off()
        self._ax.autoscale_view("tight")

        # get the artist and its colorbar
        dt = np.atleast_2d(np.linspace(0, 1, 100))
        dt = (dt.T @ dt) * 50
        self._artists["image"] = self._ax.imshow(
            dt,
            cmap=colormap,
            aspect=1,
        )
        self.add_artist(artist=self._artists["image"], name="image")

        # add the colorbar
        self.colorbar = self._fig.colorbar(
            self._artists["image"],
            ax=self._ax,
            location="bottom",
            anchor=(0.5, 1.0),
            shrink=0.66,
            fraction=0.075,
            pad=0.05,
            orientation="horizontal",
        )
        self.colorbar.minorticks_on()

        # create the hover mask
        self.hover_widget.add_label("x", "", 0)
        self.hover_widget.add_label("y", "", 0)
        self.hover_widget.add_label("temperature", "°C", 1)

        # make transparent background
        self.figure.patch.set_facecolor("None")
        self._ax.patch.set_alpha(0)
        self.setStyleSheet("background-color:transparent;")

    def update_view(self, data: np.ndarray, force: bool = False) -> None:
        """
        render the provided data.

        Parameters
        ----------
        data: 2D numpy.ndarray
            the matrix containing the temperatures collected on one sample.

        force: bool
            if True, force the redraw of the colorbar and perform a new color
            normalization.

        """
        # check the entries
        txt = "data must be a 2D array."
        assert isinstance(data, np.ndarray), txt
        assert data.ndim == 2, txt

        # update the image data
        self._artists["image"].set_data(data)
        ext = [0, data.shape[1], data.shape[0], 0]
        self._artists["image"].set_extent(ext)

        # update the colorbar data
        if np.min(data) < self.bounds[0] or force:
            self.bounds[0] = np.min(data)
        if np.max(data) > self.bounds[1] or force:
            self.bounds[1] = np.max(data)
        new_min = self.colorbar.vmin > self.bounds[0]
        new_max = self.colorbar.vmax < self.bounds[1]
        if new_min or new_max or force:

            # update the bar
            self.colorbar.set_ticks(self.bounds)
            labels = [self._colorbar_formatter.format(i) for i in self.bounds]
            self.colorbar.set_ticklabels(labels)
            self.colorbar.vmin = self.bounds[0]
            self.colorbar.vmax = self.bounds[1]

            # adjust the color normalization of the image
            norm = plc.Normalize(*self.bounds)
            self._artists["image"].set_norm(norm)

        # resize if appropriate
        new_shape = self.data.shape[0] != data.shape[0]
        new_shape |= self.data.shape[1] != data.shape[1]
        if new_shape:
            self._resize_event(None)
        self.data = data

        # update the view and the hover
        super().update_view()
        self.update_hover(self.event)

    def update_hover(self, event=None):
        """
        update the hover as required.
        """
        if event is not None:
            if event.xdata is not None and event.ydata is not None:
                x = int(round(event.xdata))
                y = int(round(event.ydata))
                t = float(self.data[y, x])
                labels = {"x": x, "y": y, "temperature": t}
                super().update_hover(event, **labels)
            else:
                self.leave_event()
        else:
            self.leave_event()


class LeptonWidget(qtw.QWidget):
    """
    Initialize a PySide2 widget capable of communicating to
    an pure thermal device equipped with a lepton 3.5 sensor.
    """

    # private variables
    _size = 50
    timer = None
    zoom_spinbox = None
    frequency_spinbox = None
    thermal_image = None
    rotation_button = None
    status_bar = None
    recording_pane = None
    opt_pane = None
    device = None

    def _create_box(self, title, obj):
        """
        create a groupbox with the given title and incorporating obj.

        Parameters
        ----------
        title: str
            the box title

        obj: QWidget
            the object to be included

        Returns
        -------
        box: QGroupBox
            the box.
        """
        # check the entries
        assert isinstance(title, str), "title must be a string."
        assert isinstance(obj, qtw.QWidget), "obj must be a QWidget."

        # generate the input layout
        layout = qtw.QHBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(obj)
        pane = qtw.QGroupBox(title)
        pane.setLayout(layout)
        pane.setFont(font)
        return pane

    def start(self):
        """
        start the timer.
        """
        try:
            self.timer.stop()
            self.timer.start(int(round(1000.0 * self.device._dt)))
        except Exception:
            pass

    def show(self):
        """
        make the widget visible.
        """
        self.device.capture(save=False)
        self.start()
        super().show()

    def update_frequency(self) -> None:
        """
        set the sampling frequency.
        """
        fs = self.frequency_spinbox.value()
        self.device.set_sampling_frequency(fs)
        self.device.interrupt()
        self.start()

    def rotate(self) -> None:
        """
        set the rotation angle.
        """
        self.device.set_angle(self.device.angle + 90)

    def rec_start(self):
        """
        function handling what happens at the start of the recording.
        """
        self.device.interrupt()
        self.device.capture(save=True)

    def rec_stop(self):
        """
        function handling what happens at the stop of the recording.
        """
        self.device.interrupt()
        if len(self.device._data) > 0:

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
                self.setCursor(qtc.Qt.WaitCursor)
                try:
                    self.device.save(path)
                    folders = path.split(os.path.sep)[:-1]
                    self.path = os.path.sep.join(folders)

                except TypeError as err:
                    msgBox = qtw.QMessageBox()
                    msgBox.setIcon(qtw.QMessageBox.Warning)
                    msgBox.setText(err)
                    msgBox.setFont(font)
                    msgBox.setWindowTitle("ERROR")
                    msgBox.setStandardButtons(qtw.QMessageBox.Ok)
                    msgBox.exec()

                # reset the camera buffer and restart the data streaming
                finally:
                    self.setCursor(qtc.Qt.ArrowCursor)
                    self.device.clear()
        else:
            msgBox = qtw.QMessageBox()
            msgBox.setIcon(qtw.QMessageBox.Warning)
            msgBox.setText("NO DATA HAVE BEEN COLLECTED.")
            msgBox.setFont(font)
            msgBox.setWindowTitle("ERROR")
            msgBox.setStandardButtons(qtw.QMessageBox.Ok)
            msgBox.exec()

        # restart sampling
        self.device.capture(save=False)
        self.start()

    def update_view(self) -> None:
        """
        update the last frame and display it.
        """
        tic = time.time()
        # NOTE: rotation is handled by LeptonCamera as it directly affects
        # the way the data are collected
        if self.device._last is not None:
            self.thermal_image.update_view(self.device._last[1])
        toc = time.time()
        fps = 0 if toc == tic else (1 / (toc - tic))
        self.status_bar.setText("FPS: {:0.1f}".format(fps))

    def __init__(self, colormap="viridis") -> None:
        """
        constructor
        """
        super().__init__()

        # actual path
        self.path = os.path.sep.join(__file__.split(os.path.sep)[:-1])

        # find the Lepton Camera device
        self.device = LeptonCamera()

        # sampling frequency
        self.frequency_spinbox = qtw.QDoubleSpinBox()
        self.frequency_spinbox.setFont(font)
        self.frequency_spinbox.setDecimals(1)
        self.frequency_spinbox.setMinimum(1.0)
        self.frequency_spinbox.setSingleStep(0.1)
        self.frequency_spinbox.setMaximum(8.7)
        self.frequency_spinbox.setValue(5.0)
        self.frequency_spinbox.valueChanged.connect(self.update_frequency)
        freq_box = self._create_box("Frequency (Hz)", self.frequency_spinbox)

        # camera rotation
        rotation_icon_path = [self.path, "_contents", "rotation.png"]
        rotation_icon_path = os.path.sep.join(rotation_icon_path)
        rotation_icon = get_QIcon(rotation_icon_path, self._size)
        self.rotation_button = qtw.QPushButton(icon=rotation_icon)
        self.rotation_button.setFlat(True)
        self.rotation_button.setFixedHeight(self._size)
        self.rotation_button.setFixedWidth(self._size)
        self.rotation_button.clicked.connect(self.rotate)
        rotation_box = self._create_box("Rotate 90°", self.rotation_button)

        # recording
        self.recording_pane = RecordingWidget()
        self.recording_pane.started.connect(self.rec_start)
        self.recording_pane.stopped.connect(self.rec_stop)
        recording_box = self._create_box("Data recording", self.recording_pane)

        # setup the options panel
        opt_pane = qtw.QWidget()
        opt_layout = qtw.QGridLayout()
        opt_layout.setSpacing(2)
        opt_layout.setContentsMargins(0, 0, 0, 0)
        opt_layout.addWidget(freq_box, 0, 0)
        opt_layout.addWidget(rotation_box, 0, 1)
        opt_layout.addWidget(recording_box, 0, 2)
        opt_pane.setLayout(opt_layout)
        opt_pane.setFixedHeight(int(round(self._size * 1.5)))

        # thermal image
        self.thermal_image = ThermalImageWidget()

        # status bar
        self.status_bar = qtw.QLabel("")
        size = font.pixelSize()
        family = font.family()
        self.status_bar.setFont(qtg.QFont(family, size // 2))
        self.status_bar.setAlignment(qtc.Qt.AlignVCenter | qtc.Qt.AlignLeft)
        self.status_bar.setFixedHeight(self.status_bar.sizeHint().height())

        # widget layout
        layout = qtw.QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.thermal_image)
        layout.addWidget(opt_pane)
        layout.addWidget(self.status_bar)
        self.setLayout(layout)
        icon = os.path.sep.join([self.path, "_contents", "main.png"])
        self.setWindowIcon(get_QIcon(icon, self._size))
        self.setWindowTitle("LeptonWidget")

        # stream handlers
        self.timer = qtc.QTimer()
        self.timer.timeout.connect(self.update_view)
        self.update_frequency()


if __name__ == "__main__":

    # highdpi scaling
    qtw.QApplication.setAttribute(qtc.Qt.AA_EnableHighDpiScaling, True)
    qtw.QApplication.setAttribute(qtc.Qt.AA_UseHighDpiPixmaps, True)

    # app generation
    app = qtw.QApplication(sys.argv)
    camera = LeptonWidget()
    camera.show()
    sys.exit(app.exec_())
