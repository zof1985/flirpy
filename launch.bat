@ECHO OFF
FOR /f %%p in ('where python') do SET PYTHONPATH=%%p
ECHO %PYTHONPATH% launcher.py