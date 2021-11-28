# lepton 3.5 purethermal camera dll imports
# the proper folder and file are defined by the __init__ file
from Lepton import CCI
from IR16Filters import IR16Capture, NewBytesFrameEvent

# python useful packages
from threading import Thread
from datetime import datetime
import numpy as np
import json
import os

# GUI-related imports
import PySide2.QtWidgets as qtw
import PySide2.QtCore as qtc
import PySide2.QtGui as qtg
import qimage2ndarray
import cv2


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
    _device = None
    _capture = None
    _data = {}
    _recording = False
    _last = None

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

    def _add_frame(self, array, width, height):
        """
        add a new frame to the buffer of readed data.
        """

        # get the sampling timestamp
        dt = datetime.now()

        # parse the thermal data to become a readable numpy array
        img = np.fromiter(array, dtype="uint16").reshape(height, width)
        img = (img - 27315.0) / 100.0  # centikelvin --> celsius conversion

        # get the recording time
        if len(self._data) > 1:
            keys = [i for i in self._data.keys()]
            delta = (dt - keys[0]).total_seconds()
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
            fps = 1.0 / (dt - self._last["timestamp"]).total_seconds()

        # update the last reading
        labels = ["timestamp", "image", "fps", "recording_time"]
        values = [dt, img, fps, lapsed]
        self._last = {i: j for i, j in zip(labels, values)}

        # update the list of collected data
        if self.is_recording():
            self._data[dt] = img

    def get_last(self):
        """
        return the last sampled data.
        """
        return self._last

    def get_shape(self):
        """
        return the shape of the collected images.
        """
        last = self.get_last()
        if last is None:
            return None
        return last["image"].shape

    @property
    def aspect_ratio(self):
        shape = self.get_shape()
        if shape is None:
            return None
        return shape[0] / shape[1]

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
        self._capture.RunGraph()
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
        self._capture.StopGraph()

    def clear(self):
        """
        clear the current object memory and buffer
        """
        self._data = {}
        self._last = None

    def to_dict(self):
        return {
            i.strftime("%d-%b-%Y %H:%M:%S.%f"): v.tolist()
            for i, v in self._data.items()
        }

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


class LeptonWidget(qtw.QWidget):
    """
    Initialize a Widget capable of visualizing videos sampled from
    an external device.
    """

    def __init__(self, gain_mode="HIGH"):
        """
        constructor
        """
        super(LeptonWidget, self).__init__()

        # camera initializer
        self._camera = LeptonCamera(gain_mode)
        self._camera.capture(save=False)

        # image label
        self.image_label = qtw.QLabel()
        self.image_label.setMouseTracking(True)
        self.image_label.installEventFilter(self)

        # button bar with both recording and exit button
        self.quit_button = qtw.QPushButton("QUIT")
        self.quit_button.clicked.connect(self.close)
        self.rec_button = qtw.QPushButton("● START RECORDING", self)
        self.rec_button.clicked.connect(self.record)
        self.rec_button.setCheckable(True)
        button_layout = qtw.QHBoxLayout()
        button_layout.addWidget(self.rec_button)
        button_layout.addWidget(self.quit_button)
        button_pane = qtw.QWidget()
        button_pane.setFixedHeight(100)
        button_pane.setLayout(button_layout)

        # temperatures label
        self.mouse_data_label = qtw.QLabel("Pointer t:  °C")
        self.mean_data_label = qtw.QLabel("Mean t:  °C")
        self.max_data_label = qtw.QLabel("Max t:  °C")
        self.min_data_label = qtw.QLabel("Min t:  °C")
        self.fps_label = qtw.QLabel("fps:")
        data_layout = qtw.QHBoxLayout()
        data_layout.addWidget(self.mouse_data_label)
        data_layout.addWidget(self.mean_data_label)
        data_layout.addWidget(self.min_data_label)
        data_layout.addWidget(self.max_data_label)
        data_layout.addWidget(self.fps_label)
        data_pane = qtw.QWidget()
        data_pane.setFixedHeight(100)
        data_pane.setLayout(data_layout)

        # main layout
        self.main_layout = qtw.QVBoxLayout()
        self.main_layout.addWidget(self.image_label)
        self.main_layout.addWidget(data_pane)
        self.main_layout.addWidget(button_pane)
        self.setLayout(self.main_layout)
        self.setWindowTitle("LeptonWidget")

        # stream handler
        self._timer = qtc.QTimer()
        self._timer.timeout.connect(self.stream_video)
        self._timer.start(100)

    def eventFilter(self, source, event):

        if self._camera.get_last() is not None:

            # update the min temperature label
            min_temp = np.min(self._camera.get_last()["image"])
            min_txt = "Min t: {:0.1f} °C".format(min_temp)
            self.min_data_label.setText(min_txt)

            # update the max temperature label
            max_temp = np.max(self._camera.get_last()["image"])
            max_txt = "Max t: {:0.1f} °C".format(max_temp)
            self.max_data_label.setText(max_txt)

            # update the avg temperature label
            avg_temp = np.mean(self._camera.get_last()["image"])
            avg_txt = "Mean t: {:0.1f} °C".format(avg_temp)
            self.mean_data_label.setText(avg_txt)

            # update the fps label
            fps_txt = "fps: {:0.1f}".format(self._camera.get_last()["fps"])
            self.fps_label.setText(fps_txt)

            # check if the pointer is on the image and update pointer temperature
            if event.type() == qtc.QEvent.MouseMove:

                # get the mouse coordinates
                x, y = (event.x(), event.y())

                # rescale to the original image size
                shape = self._camera.get_shape()
                w_res = int(x * shape[1] / self.image_label.width())
                h_res = int(y * shape[0] / self.image_label.height())

                # update data_label with the temperature at mouse position
                temp = self._camera.get_last()["image"][h_res, w_res]
                txt = "Pointer t: {:0.1f} °C".format(temp)
                self.mouse_data_label.setText(txt)

            # the pointer leaves the image, therefore no temperature has to be shown
            elif event.type() == qtc.QEvent.Leave:
                txt = "Pointer t:  °C"
                self.mouse_data_label.setText(txt)

        return False

    def stream_video(self):
        """
        display the last captured images.
        """
        if self._camera.get_last() is not None:

            # get the image
            img = self._camera.get_last()["image"]

            # convert to grayscale
            gry = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
            gry = np.expand_dims(gry, 2).astype(np.uint8)
            gry = cv2.merge([gry, gry, gry])

            # resize preserving the aspect ratio
            # view_w = self.image_label.width()
            # view_h = self.image_label.height()
            h = img.shape[0] * 5
            w = img.shape[1] * 5
            resized_image = cv2.resize(gry, (w, h))

            # converto to heatmap
            heatmap = cv2.applyColorMap(resized_image, cv2.COLORMAP_HOT)

            # set the recording overlay if required
            if self._camera.is_recording():
                cv2.putText(
                    heatmap,
                    "REC: {}".format(self._camera.get_last()["recording_time"]),
                    (10, int(h * 0.95)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    2,
                )

            # update the view
            qimage = qimage2ndarray.array2qimage(heatmap)
            self.image_label.setPixmap(qtg.QPixmap.fromImage(qimage))
            self.setFixedSize(self.size())

    def record(self):
        """
        start and stop the recording of the data.
        """
        if self.rec_button.isChecked():
            self.rec_button.setText("■ STOP RECORDING")
            self._camera._recording = True
        else:
            self.rec_button.setText("● START RECORDING")
            self._camera._recording = False
            if len(self._camera._data) > 0:

                # stop the timer
                self._timer.stop()

                # let the user decide where to save the data
                path = qtw.QFileDialog.getSaveFileName(
                    self, "json / npz file dialog", os.getcwd()
                )

                # save the data
                if len(path) > 0:
                    path = path[0].replace("/", os.path.sep)
                    ext = path.split(".")[-1]
                    if ext == path:
                        ext = "json"
                        path += ".json"
                    root = os.path.sep.join(path.split(os.path.sep)[:-1])
                    os.makedirs(root, exist_ok=True)

                    # check the file extension and type
                    if ext.lower() == "json":
                        self._camera.to_json(path)
                    elif ext.lower() == "npz":
                        self._camera.to_npz(path)
                    else:
                        raise NotImplementedError(
                            "{} extension not supported.".format(ext)
                        )

                    # reset the camera buffer
                    self._camera.clear()

                # restart the timer
                self._timer.start()
