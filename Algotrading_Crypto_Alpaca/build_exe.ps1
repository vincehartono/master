@echo off
REM Build Alpaca Trading Bot EXE
REM Run this script to create AlpacaTradingBot.exe

echo [*] Building Alpaca Crypto Trading Bot EXE...
echo.

REM Check if PyInstaller is installed
pip show pyinstaller >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] PyInstaller not found. Installing...
    pip install pyinstaller
)

REM Build the EXE
echo [+] Building EXE from trade.spec...
pyinstaller trade.spec --onefile --distpath dist

echo.
if exist "dist\AlpacaTradingBot.exe" (
    echo [OK] Build successful!
    echo [+] EXE location: dist\AlpacaTradingBot.exe
    echo.
    echo [*] To run the bot:
    echo     1. Copy your .env file to the same folder as AlpacaTradingBot.exe
    echo     2. Double-click AlpacaTradingBot.exe
    echo     3. Open http://localhost:5000 in your browser
) else (
    echo [ERROR] Build failed
)

pause
