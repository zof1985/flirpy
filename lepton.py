# lepton 3.5 purethermal camera dll imports
# the proper folder and file are defined by the __init__ file
from Lepton import CCI
from IR16Filters import IR16Capture, NewBytesFrameEvent

# python useful packages
from threading import Thread
from datetime import datetime
import numpy as np
import h5py
import json
import os


class LeptonCamera:
    """
    Initialize a Lepton camera object capable of communicating to
    an pure thermal device equipped with a lepton 3.5 sensor.

    Parameters
    ----------
    gain_mode: str
        any between "LOW" or "HIGH". It defines the gain mode of the lepton camera.
    """

    # class variables
    device = None
    reader = None
    _data = {}
    _recording = False
    _last = None
    time_format = "%d-%b-%Y %H:%M:%S.%f"

    def __init__(self, gain_mode="HIGH"):
        """
        constructor
        """
        super(LeptonCamera, self).__init__()
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
                    self.device = devices[idx]
                    break
                else:
                    print("Unrecognized input value.\n")

        # if just one device is found, select it
        elif len(devices) == 1:
            self.device = devices[0]

        # tell the user that no valid devices have been found.
        else:
            self.device = None

        # open the found device
        txt = "No devices called 'PureThermal' have been found."
        assert self.device is not None, txt
        self.device = self.device.Open()
        self.device.sys.RunFFCNormalization()

        # set the gain mode
        self.set_gain(gain_mode)

        # set radiometric
        try:
            self.device.rad.SetTLinearEnableStateChecked(True)
        except:
            print("this lepton does not support tlinear")

        # setup the buffer
        self.reader = IR16Capture()
        self.reader.SetupGraphWithBytesCallback(NewBytesFrameEvent(self._add_frame))

    def get_gain(self):
        """
        return the actual gain mode.
        """
        return self.device.sys.GetGainMode()

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
                self.device.sys.SetGainMode(CCI.Sys.GainMode.HIGH)
            else:
                self.device.sys.SetGainMode(CCI.Sys.GainMode.LOW)
        else:
            valid_gains = [CCI.Sys.GainMode.HIGH, CCI.Sys.GainMode.LOW]
            txt += " or any of {}".format(valid_gains)
            assert gain_mode in valid_gains, txt
            self.device.sys.SetGainMode(gain_mode)

    def _add_frame(self, array, width, height):
        """
        add a new frame to the buffer of readed data.
        """

        # get the sampling timestamp
        dt = datetime.now()
        timestamp = dt.strftime(self.time_format)

        # parse the thermal data to become a readable numpy array
        img = np.fromiter(array, dtype="uint16").reshape(height, width)
        img = (img - 27315.0) / 100.0  # centikelvin --> celsius conversion
        img = img.astype(np.float16)

        # get the recording time
        if len(self._data) > 1:
            keys = [i for i in self._data.keys()]
            t0 = datetime.strptime(keys[0], self.time_format)
            delta = (dt - t0).total_seconds()
            h = int(delta // 3600)
            m = int((delta - h * 3600) // 60)
            s = int((delta - h * 3600 - m * 60) // 1)
            d = int(((delta % 1) * 1e6) // 1e5)
            lapsed = "{:02d}:{:02d}:{:02d}.{:01d}".format(h, m, s, d)
        else:
            lapsed = "00:00:00.0"

        # get the fps
        if self._last is None:
            fps = 0.0
        else:
            t1 = datetime.strptime(self._last["timestamp"], self.time_format)
            fps = 1.0 / (dt - t1).total_seconds()

        # update the last reading
        labels = ["timestamp", "image", "fps", "recording_time"]
        values = [timestamp, img, fps, lapsed]
        self._last = {i: j for i, j in zip(labels, values)}

        # update the list of collected data
        if self.is_recording():
            self._data[timestamp] = img.astype

    def get_last(self):
        """
        return the last sampled data.
        """
        return self._last

    def get_shape(self):
        """
        return the shape of the collected images.
        """
        return (120, 160)

    @property
    def aspect_ratio(self):
        shape = self.get_shape()
        if shape is None:
            return None
        return shape[1] / shape[0]

    def is_recording(self):
        return self._recording

    def capture(self, save=True, n_frames=None, time=None):
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
        self.reader.RunGraph()
        while self.get_last() is None:
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

    def stop(self):
        """
        stop reading from camera.
        """
        self._recording = False
        self.reader.StopGraph()

    def clear(self):
        """
        clear the current object memory and buffer
        """
        self._data = {}
        self._last = None

    def to_dict(self):
        """
        return the sampled data as dict with
        timestamps as keys and the sampled data as values.

        Returns
        -------
        d: dict
            the dict containing the sampled data.
            Timestamps are provided as keys and the sampled data as values.
        """
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
        t = np.array(list(self._data.keys()))
        x = np.atleast_3d(list(self._data.values()), dtype=np.float16)
        return t, x

    def to_npz(self, filename):
        """
        store the recorded data to a compressed npz file.

        Parameters
        ----------
        filename: str
            a valid filename path
        """
        timestamps, images = self.to_numpy()
        np.savez(filename, timestamps=timestamps, images=images)

    def to_json(self, filename):
        """
        store the data as a json file.

        Parameters
        ----------
        filename: str
            a valid filename path
        """
        with open(filename, "w") as buf:
            json.dump(self.to_dict(), buf)

    def to_h5(self, filename, compression=9):
        """
        store the data as a (gzip compressed) h5 file.

        Parameters
        ----------
        filename: str
            a valid filename path

        compression: int
            the h5 compression level (default 9)
        """
        hf = h5py.File(filename, "w")
        times, samples = self.to_numpy()
        hf.create_dataset(
            "timestamps",
            data=times,
            compression="gzip",
            compression_opts=compression,
        )
        hf.create_dataset(
            "samples", data=samples, compression="gzip", compression_opts=compression
        )
        hf.close()
