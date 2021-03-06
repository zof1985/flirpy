# imports
import clr  # needs the "pythonnet" package
import sys
import os
import platform

# check whether python is running as 64bit or 32bit
# to import the right .NET dll
folder = "x64" if platform.architecture()[0] == "64bit" else "x86"
path = os.path.sep.join(__file__.split(os.path.sep)[:-1])
sys.path.append(os.path.sep.join([path, folder]))
clr.AddReference("LeptonUVC")
clr.AddReference("ManagedIR16Filters")

# generic imports
from .lepton import *
