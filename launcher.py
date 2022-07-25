# imports
import clr  # needs the "pythonnet" package
import sys
import platform
import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qtw

# check whether python is running as 64bit or 32bit
# to import the right .NET dll
folder = "x64" if platform.architecture()[0] == "64bit" else "x86"
sys.path.append(folder)
clr.AddReference("LeptonUVC")
clr.AddReference("ManagedIR16Filters")

# generic imports
from lepton import *


if __name__ == "__main__":

    # highdpi scaling
    qtw.QApplication.setAttribute(qtc.Qt.AA_EnableHighDpiScaling, True)
    qtw.QApplication.setAttribute(qtc.Qt.AA_UseHighDpiPixmaps, True)

    # app generation
    app = qtw.QApplication(sys.argv)
    camera = LeptonWidget()
    camera.show()
    sys.exit(app.exec_())