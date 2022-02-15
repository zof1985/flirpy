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
from flirpy import LeptonCamera

camera = LeptonCamera(sampling_frequency=5) # inizialize the camera object
camera.capture(n_frames=10) # capture 10 frames
timestamps, images = camera.to_numpy()  # obtain the outcomes as numpy arrays.
for time, img in zip(timestamps, images):
    plt.imshow(img)
    plt.title(time)
    plt.show()
```

### Example 2 - GUI with recording and visualization options

```
# imports
from flirpy import LeptonCameraWidget
import PySide2.QtWidgets as qtw
import sys

# within the main create a pyside2 QApplication
if __name__ == "__main__":

    # enble High DPI screen scaling (optional)
    qtw.QApplication.setAttribute(qtc.Qt.AA_EnableHighDpiScaling, True)
    qtw.QApplication.setAttribute(qtc.Qt.AA_UseHighDpiPixmaps, True)

    # app generation
    app = qtw.QApplication(sys.argv)
    camera = LeptonCameraWidget(sampling_frequency=5) # inizialize the camera object
    camera.show()
    sys.exit(app.exec_())
```

---
