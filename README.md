# FLIRPY

python integration for FLIR Lepton 3.5

## Getting started

1. Make sure to unlock the SDK dlls:

   * Navigate to x64 or x86 depending on if you have 64 bit (x64) or 32 bit (x86) python.
   * Right click on LeptonUVC.dll and select Properties.
   * In the general tab there may be a section called "Security" at the bottom. If there is, check "Unblock" and hit apply.
   * Repeat for ManagedIR16Filters.dll.
2. Install the dependencies:

```
pip install -r requirements.txt
```

3. Connect a purethermal board to your PC.

## Usage

### Example 1 - sensor-only

```
# use only the sensor to capture some data
import matplotlib.pyplot as plt
from flirpy import *
from time import sleep
from numpy.random import permutation
import os

# initialize the camera object
camera = LeptonCamera()

# capture 10 frames
camera.capture(save=True, n_frames=10)

# save the frames in "h5 format"
frames_file = os.path.sep.join([os.getcwd(), "frames.h5"])
camera.save(frames_file)
frame_readings = read_file(frames_file)

# capture 2 seconds with sampling frequency set at 3Hz
camera.set_sampling_frequency(3)
camera.clear()  # ensure to free the stored data buffer
camera.capture(save=True, seconds=2)

# save the collected data in "npz format"
seconds_file = os.path.sep.join([os.getcwd(), "seconds.npz"])
camera.save(seconds_file)
seconds_readings = read_file(seconds_file)

# collect data for a random amount of time in "json format"
camera.clear()
camera.set_sampling_frequency(4.5)
camera.capture(save=True)
sleep(permutation([1.5, 1.0, 0.8])[0])
camera.interrupt()
random_file = os.path.sep.join([os.getcwd(), "random.json"])
camera.save(random_file)
random_readings = read_file(random_file)

# show the data contained in the camera buffer
timestamps, images = camera.to_numpy()
fig, ax = plt.subplots(len(timestamps), 1)
for i in range(len(timestamps)):
    ax[i].imshow(to_heatmap(images[0]))
    ax[i].set_title(timestamps[i].strftime(LeptonCamera.date_format()))
plt.tight_layout()
plt.show()
```

### Example 2 - GUI with recording and visualization options

```
# imports
from flirpy import *
import PySide2.QtWidgets as qtw
import PySide2.QtCore as qtc
import sys

if __name__ == "__main__":

    # highdpi scaling
    qtw.QApplication.setAttribute(qtc.Qt.AA_EnableHighDpiScaling, True)
    qtw.QApplication.setAttribute(qtc.Qt.AA_UseHighDpiPixmaps, True)

    # app generation
    app = qtw.QApplication(sys.argv)
    camera = LeptonCameraWidget()
    camera.show()
    sys.exit(app.exec_())
```

---
