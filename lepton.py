# imports
from Lepton import CCI
from typing import Tuple
from scipy import ndimage
from datetime import datetime
from IR16Filters import IR16Capture, NewBytesFrameEvent

import PySide2.QtWidgets as qtw
import PySide2.QtCore as qtc
import PySide2.QtGui as qtg
import numpy as np
import threading
import h5py
import json
import time
import os

# matplotlib options
from matplotlib import use as matplotlib_use

matplotlib_use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.pyplot as plt
import matplotlib.colors as plc

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


class ThermalHoverWidget(qtw.QWidget):
    """
    defines a thermal image hover table object.
    """

    # class variable
    fps = None
    fps_format = "{:0.2f}"
    coords = None
    coords_format = "({:0.0f},{:0.0f})"
    temp = None
    temp_format = "{:0.2f}°C"

    def __init__(self):
        super().__init__()
        layout = qtw.QGridLayout()

        # add the labels
        for i, v in enumerate(["FPS:", "(X,Y):", "Temperature:"]):
            lab = qtw.QLabel(v)
            lab.setAlignment(qtc.Qt.AlignVCenter | qtc.Qt.AlignLeft)
            lab.setFont(font)
            layout.addWidget(lab, i, 0)

        # add the fps
        self.fps = qtw.QLabel("")
        self.fps.setAlignment(qtc.Qt.AlignVCenter | qtc.Qt.AlignRight)
        self.fps.setFont(font)
        layout.addWidget(self.fps, 0, 1)

        # add coords
        self.coords = qtw.QLabel("")
        self.coords.setAlignment(qtc.Qt.AlignVCenter | qtc.Qt.AlignRight)
        self.coords.setFont(font)
        layout.addWidget(self.coords, 1, 1)

        # add temperature
        self.temp = qtw.QLabel("")
        self.temp.setAlignment(qtc.Qt.AlignVCenter | qtc.Qt.AlignRight)
        self.temp.setFont(font)
        layout.addWidget(self.temp, 2, 1)

        # setup the layout
        self.setLayout(layout)
        flags = qtc.Qt.FramelessWindowHint | qtc.Qt.WindowStaysOnTopHint
        self.setWindowFlags(flags)

    def update(self, fps=None, x=None, y=None, temp=None):
        """
        update the hover parameters.

        Parameters
        ----------
        fps: float
            the fps value

        x: int
            the x coordinate

        y: int
            the y coordinate

        temp: float
            the temperature value
        """
        if fps is not None:
            self.fps.setText(self.fps_format.format(fps))
        if x is not None and y is not None:
            self.coords.setText(self.coords_format.format(x, y))
        if temp is not None:
            self.temp.setText(self.temp_format.format(temp))


class FigureWidget(FigureCanvasQTAgg):
    """
    Generate a QWidget incorporating a matplotlib Figure.
    Animated artists can be provided to ensure optimal performances.

    Parameters
    ----------
    figure: matplotlib Figure
        the figure to be rendered.

    artists: list
        the list of artists that have to be animated.
    """

    # class variables
    figure = None
    animated_artists = {}
    _background = None
    _cid = None
    _res = None

    def __init__(self, figure, animated_artists={}):
        super().__init__(figure)
        self.figure = figure
        self._background = None
        self.add_artists(**animated_artists)
        self._res = self.figure.canvas.mpl_connect(
            "resize_event",
            self._resize_event,
        )
        self._cid = self.figure.canvas.mpl_connect(
            "draw_event",
            self.on_draw,
        )

    def add_artists(self, **kwargs):
        """
        add animated artists to the animator.
        """
        for key, value in kwargs.items():
            value.set_animated(True)
            self.animated_artists[key] = value
            self.figure.add_artist(value)

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
        for a in self.animated_artists.values():
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

    def _resize_event(self, event):
        """
        handle object resizing.
        """
        self.figure.tight_layout()
        self.figure.canvas.draw()


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
    _tick_formatter = "{:0.1f}°C"
    bounds = [1e5, -1e5]

    def __init__(self, colormap: str = "viridis") -> None:

        # generate the figure and axis
        fig, ax = plt.subplots(1, 1, dpi=300)
        ax.set_axis_off()  # remove all axes

        # get the artist and its colorbar
        dt = np.atleast_2d(np.linspace(0, 1, 100))
        dt = (dt.T @ dt) * 50
        art = ax.imshow(dt, cmap=colormap, aspect=1)

        # store the data
        super().__init__(fig, {"image": art})
        self.axis = ax

        # add the colorbar
        cb = fig.colorbar(
            art,
            ax=ax,
            location="bottom",
            anchor=(0.5, 1.0),
            shrink=0.66,
            fraction=0.075,
            pad=0.05,
            orientation="horizontal",
        )
        cb.minorticks_on()
        self.colorbar = cb

        # create the hover mask
        self.hover_widget = ThermalHoverWidget()
        self.hover_widget.setVisible(False)

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

    def update_view(self, data: np.ndarray, force=False) -> None:
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
        self.animated_artists["image"].set_data(data)
        ext = [0, data.shape[1], data.shape[0], 0]
        self.animated_artists["image"].set_extent(ext)

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
            labels = [self._tick_formatter.format(i) for i in self.bounds]
            self.colorbar.set_ticklabels(labels)
            self.colorbar.vmin = self.bounds[0]
            self.colorbar.vmax = self.bounds[1]

            # adjust the color normalization of the image
            norm = plc.Normalize(*self.bounds)
            self.animated_artists["image"].set_norm(norm)

        # resize if appropriate
        new_shape = self.data.shape[0] != data.shape[0]
        new_shape |= self.data.shape[1] != data.shape[1]
        if new_shape:
            self._resize_event(None)
        self.data = data

        # update the hover data
        if self.event is not None:
            self.update_hover()

        super().update_view()

    def update_hover(self):
        """
        update the hover as required.
        """
        try:
            # get the temperature
            x = int(round(self.event.xdata))
            y = int(round(self.event.ydata))
            t = self.data[y, x]

            # update the fps
            new = time.time()
            delta = new - self._old
            fps = (1 / delta) if delta > 0 else 0
            self._old = new
            self.hover_widget.update(fps, x, y, t)

            # adjust the hover position
            xmax, ymax = self.event.canvas._lastKey[:2]
            y = ymax - self.event.y
            x = self.event.x
            x_off = int(round(xmax * 0.05))
            y_off = int(round(ymax * 0.05))
            pnt = self.mapToGlobal(qtc.QPoint(x + x_off, y + y_off))
            self.hover_widget.move(pnt.x(), pnt.y())

        except Exception:
            self.leave_event()

    def enter_event(self, event=None):
        """
        handle the entry of the mouse over the area.
        """
        self.move_event(event)

    def leave_event(self, event=None):
        """
        handle the entry of the mouse over the area.
        """
        self.hover_widget.setVisible(False)
        self.event = None

    def move_event(self, event=None):
        """
        handle the movement of the mouse over the area.
        """
        if not self.hover_widget.isVisible():
            self.hover_widget.setVisible(True)
        self.event = event
        self.update_hover()


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
        self.start()

    def update_view(self) -> None:
        """
        update the last frame and display it.
        """
        # NOTE: rotation is handled by LeptonCamera as it directly affects
        # the way the data are collected
        if self.device._last is not None:
            self.thermal_image.update_view(self.device._last[1])

    def __init__(self) -> None:
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
        self.opt_pane = qtw.QWidget()
        opt_layout = qtw.QHBoxLayout()
        opt_layout.setSpacing(2)
        opt_layout.setContentsMargins(0, 0, 0, 0)
        opt_layout.addWidget(freq_box)
        opt_layout.addWidget(rotation_box)
        opt_layout.addWidget(recording_box)
        self.opt_pane.setLayout(opt_layout)
        self.opt_pane.setFixedHeight(int(round(self._size * 1.5)))

        # thermal image
        self.thermal_image = ThermalImageWidget()

        # widget layout
        layout = qtw.QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.thermal_image)
        layout.addWidget(self.opt_pane)
        self.setLayout(layout)
        icon = os.path.sep.join([self.path, "_contents", "main.png"])
        self.setWindowIcon(get_QIcon(icon, self._size))
        self.setWindowTitle("LeptonWidget")

        # stream handlers
        self.timer = qtc.QTimer()
        self.timer.timeout.connect(self.update_view)
        self.update_frequency()
