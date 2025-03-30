@echo off
cd /d "%~dp0"
call C:\Users\marti\anaconda3\Scripts\conda.exe init
call C:\Users\marti\anaconda3\Scripts\conda.exe activate SDAPROJECTS   REM Replace 'myenv' with your conda environment name
call C:\Users\marti\anaconda3\envs\SDAPROJECTS\python gui_main.py
pause