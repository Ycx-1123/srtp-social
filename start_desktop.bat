@echo off
setlocal
cd /d "%~dp0"
if defined SOCI_PYTHON (
    "%SOCI_PYTHON%" "%~dp0desktop_launcher.py" %*
) else (
    python "%~dp0desktop_launcher.py" %*
)
endlocal
