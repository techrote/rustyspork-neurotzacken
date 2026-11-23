@echo off
setlocal
set "PYTHONPATH=%~dp0;%PYTHONPATH%"
python -m rustyspork %*
exit /b %errorlevel%
