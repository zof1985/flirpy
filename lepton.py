# dll imports
# the proper folder and file are defined by the __init__ file
from Lepton import CCI
from IR16Filters import IR16Capture, NewBytesFrameEvent

# additional imports
from collections import deque
from threading import Thread
from datetime import datetime, timedelta
import numpy as np


class Lepton:
    """
    Initialize a Lepton camera object able to capture thermal images from
    a lepton 3.5 sensor.

    Parameters
    ----------
    sampling_frequency: float/int
        the selected sampling frequency. It must be a float or int value
        in the (0, 8.7] range.

    shape: list, tuple of int with len = 2
        the shape of the resulting image in pixels.

    gain_mode: str
        any between "LOW" or "HIGH". It defines the gain mode of the lepton camera.
    """

    # class variables
    _device = None
    _buffer = deque()
    _capture = None
    _data = {}
    _recording = False
    _shape = (120, 120)
    _sampling_frequency = 8.7
    _read_thread = None
    _n_frames = None
    _time = None

    # constructor
    def __init__(self, sampling_frequency=8.7, shape=(120, 120), gain_mode="HIGH"):

        # find a valid device
        devices = [i for i in CCI.GetDevices() if i.Name.startswith("PureThermal")]

        # if multiple devices are found, allow the user to select the preferred one
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
        self.set_gain(gain_mode)

        # set the sampling frequency
        self.set_sampling_frequency(sampling_frequency)

        # set the image output shape
        self.set_shape(shape)

        # set radiometric
        try:
            self._device.rad.SetTLinearEnableStateChecked(True)
        except:
            print("this lepton does not support tlinear")

        # setup the buffer
        self._capture = IR16Capture()
        self._capture.SetupGraphWithBytesCallback(NewBytesFrameEvent(self._add_frame))

    def get_gain(self):
        """
        return the actual gain mode.
        """
        return self._device.sys.GetGainMode()

    def set_gain(self, gain_mode):
        """
        set the gain mode of the current device

        Parameters
        ----------
        gain_mode: str or GainMode enum
            any between "LOW", "HIGH" or
            CCI.Sys.GainMode.HIGH, CCI.Sys.GainMode.LOW.
        """
        txt = "gain_mode must be a string (either 'LOW' or 'HIGH')"
        if isinstance(gain_mode, str):
            assert gain_mode.upper() in ["HIGH", "LOW"], txt
            if gain_mode == "HIGH":
                self._device.sys.SetGainMode(CCI.Sys.GainMode.HIGH)
            else:
                self._device.sys.SetGainMode(CCI.Sys.GainMode.LOW)
        else:
            valid_gains = [CCI.Sys.GainMode.HIGH, CCI.Sys.GainMode.LOW]
            txt += " or any of {}".format(valid_gains)
            assert gain_mode in valid_gains, txt
            self._device.sys.SetGainMode(gain_mode)

    def set_sampling_frequency(self, freq):
        """
        set the sampling frequency of the lepton camera

        Parameters
        ----------
        freq: float/int
            any value within the (0, 8.7] range.
        """

        # set the sample size
        txt = "freq must be an int or float in the (0, 8.7] range."
        assert isinstance(freq, (int, float)), txt
        assert freq > 0 and freq <= 8.7, txt
        self._sampling_frequency = freq

    def get_sampling_frequency(self):
        """
        return the actual sampling frequency.
        """
        return self._sampling_frequency

    def set_shape(self, shape):
        """
        set the shape of the collected data.

        Parameters
        ----------
        shape: tuple/list of len = 2
            a tuple of int with len = 2. The maximum input shape must be:
                width = 160 px
                height = 120 px
        """
        txt = "shape must be a tuple/list of type (int, int)."
        assert isinstance(shape, (tuple, list)), txt
        assert len(shape) == 2, txt
        assert all([isinstance(i, int) for i in shape]), txt
        txt = "width must be in the (0, {}] range."
        assert shape[0] > 0 and shape[0] <= 160, txt.format(160)
        assert shape[1] > 0 and shape[0] <= 120, txt.format(120)
        self._shape = tuple(shape)

    def get_shape(self):
        """
        return the actual input shape.
        """
        return self._shape

    def _add_frame(self, array, width, height):
        """
        add a new frame to the buffer of readed data.
        """
        self._buffer.append((height, width, array))

    def is_recording(self):
        return self._recording

    def read(self, n_frames=None, time=None):
        """
        read a series of frames from the camera.

        Parameters
        ----------
        n_frames: None / int
            if a positive int is provided, n_frames are captured.
            Otherwise, all the frames collected are saved until the
            stop command is given.

        time: None / int
            if a positive int is provided, data is sampled for the indicated
            amount of seconds.
        """
        self._recording = True
        self._buffer.clear()
        self._n_frames = n_frames
        self._time = time

        # adjust the n_frames to the sampling frequency
        if self._time is None and self._n_frames is not None:
            time = self._n_frames / self._sampling_frequency

        # start reading data
        self._capture.RunGraph()
        while len(self._buffer) == 0:
            pass
        tic = datetime.now()

        # continue reading until a stopping criterion is met
        toc = datetime.now()
        while self.is_recording():
            toc = datetime.now()
            time_delta = (toc - tic).total_seconds()
            if time is not None and time_delta >= time:
                self._recording = False
        self._capture.StopGraph()

    def parse_data(self):
        # extract the samples according to the selected sampling frequency
        n = len(self._buffer)
        time_array = np.linspace(tic.timestamp(), toc.timestamp(), n)
        time_array = [datetime.fromtimestamp(i) for i in time_array]
        dt = 1.0 / self._sampling_frequency
        sec = int(dt // 1)
        mic = int((dt % max(1, sec)) * 1e6)
        dt = timedelta(seconds=sec, microseconds=mic)
        t = time_array[0] - dt
        while t < time_array[-1]:
            t += dt
            i = 0
            while i < len(time_array) and time_array[i] < t:
                i += 1
            if i < len(time_array):
                img = self._parse_image(self._buffer[i])
                tm = time_array[i]
                self._data[tm] = img

    def read(self, n_frames=None, time=None):
        """
        read a series of frames from the camera.

        Parameters
        ----------
        n_frames: None / int
            if a positive int is provided, n_frames are captured.
            Otherwise, all the frames collected are saved until the
            stop command is given.

        time: None / int
            if a positive int is provided, data is sampled for the indicated
            amount of seconds.
        """
        self._read_thread = Thread(
            target=self._read_fun, kwargs={"n_frames": n_frames, "time": time}
        )
        self._read_thread.run()

    def stop(self):
        """
        stop reading from camera.
        """
        self._recording = False

    def _parse_image(self, img):
        """
        parse an image according to the required shape.

        Parameters
        ----------
        img: tuple
            a tuple containing:
            height, width, and data of the image.

        Returns
        -------
        parsed: a numpy 2D array of float with shape = input_shape
        """

        # adjust the output shape
        h, w, f = img
        img = np.fromiter(f, dtype="uint16").reshape(h, w)
        y_off = max(h - self._shape[0], 0) // 2
        y_idx = np.arange(y_off, y_off + self._shape[0])
        x_off = max(w - self._shape[1], 0) // 2
        x_idx = np.arange(x_off, x_off + self._shape[1])
        out = img[y_idx, :][:, x_idx]

        # the image is in centikelvin. Therefore convert it to celsius units
        return (out - 27315.0) / 100.0

    def to_dict(self):
        return self._data

    def to_numpy(self):
        """
        return the sampled data as numpy arrays.

        Returns
        -------
        t: 1D numpy array
            a numpy array containing the datetime object

        x: 3D numpy array
            a 3D array where each the first dimension correspond to each sample.
        """
        t = np.array(list(self._data.keys()), dtype=np.datetime64)
        x = np.atleast_3d(list(self._data.values()))
        return t, x

    def to_npz(self, filename):
        """
        store the recorded data to a compresse npz file.

        Parameters
        ----------
        filename: str
            a valid filename path
        """
        timestamps, images = self.to_numpy()
        np.savez(filename, timestamps=timestamps, images=images)

    '''
    def to_clip(self, filename, codec="mp4v"):
        """
        store the recorded videoclip.

        Parameters
        ----------
        filename: str
            a valid filename path

        codec: str
            a supported codec
        """

        # initialize the writer
        out = cv2.VideoWriter(
            filename,
            cv2.VideoWriter_fourcc(*codec),
            self._sampling_frequency,
            self._shape,
            False,
        )

        # get the images
        imgs = self.to_numpy()[1]

        # convert to grayscale
        grys = (imgs - np.min(imgs)) / (np.max(imgs) - np.min(imgs))
        grys = (grys * 255).astype(np.int16)

        # flip the image up-down
        for g in grys:
            out.write(g)

        # release the writer (i.e. write the file)
        out.release()
    '''
