@echo off
echo Select which executable to build:
echo   1) Trading algo (Algotrading_FX\algo.py)
echo   2) 6+ Poker GUI (6+ Poker\main.py)
set /p choice=Enter 1 or 2 (or anything else to cancel): 

if "%choice%"=="1" (
    cd /d "%~dp0Algotrading_FX"
    echo Building algo.exe ...
    pyinstaller --onefile algo.py
    echo.
    echo Done. Check dist\algo.exe in Algotrading_FX.
    goto :eof
)

if "%choice%"=="2" (
    cd /d "%~dp06+ Poker"
    echo Building main.exe (6+ Poker GUI) ...
    pyinstaller --onefile --windowed main.py
    echo.
    echo Done. Check dist\main.exe in 6+ Poker.
    goto :eof
)

echo Cancelled.

