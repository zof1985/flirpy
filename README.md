# FLIRPY

python integration for FLIR Lepton 3.5


## Getting started

1. Install the dependencies:

		pip install pythonnet numpy opencv pyside2

2. Connect a purethermal board to your PC

4. Make sure to unblock the SDK dlls:

    * Navigate to x64 or x86 depending on if you have 64 bit (x64) or 32 bit (x86) python
    * Right click on LeptonUVC.dll and select Properties
    * In the general tab there may be a section called "Security" at the bottom. If there is, check "Unblock" and hit apply. 
    * Repeat for ManagedIR16Filters.dll

## Usage

    from flirpy import Lepton
    from time import sleep

    camera = Lepton()  # inizialize the camera object
    camera.read()  # start reading data
    sleep(5)
    camera.stop() # stop reading
    timestamps, images = camera.to_numpy()  # obtain the outcomes as numpy arrays.
