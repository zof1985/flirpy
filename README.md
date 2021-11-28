# FLIRPY

python integration for FLIR Lepton 3.5


## Getting started

1. Install the dependencies:

		pip install pythonnet numpy opencv pyside2 qimage2ndarray

2. Connect a purethermal board to your PC

4. Make sure to unblock the SDK dlls:

    * Navigate to x64 or x86 depending on if you have 64 bit (x64) or 32 bit (x86) python
    * Right click on LeptonUVC.dll and select Properties
    * In the general tab there may be a section called "Security" at the bottom. If there is, check "Unblock" and hit apply.
    * Repeat for ManagedIR16Filters.dll

## Usage

    # use only the sensor to capture some data
    import matplotlib.pyplot as plt
    from flirpy import LeptonCamera

    camera = LeptonCamera()  # inizialize the camera object
    camera.capture(n_frames=10)  # read 10 frames
    timestamps, images = camera.to_numpy()  # obtain the outcomes as numpy arrays.
    for time, img in zip(timestamps, images):
        plt.imshow(img)
        plt.title(time)
        plt.show()

    # use the pyqt interface to have a real-time monitoring of the data

    from flirpy.lepton import LeptonWidget
    from PySide2.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    camera = LeptonWidget()
    camera.show()
    sys.exit(app.exec_())
