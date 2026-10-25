@echo off
setlocal
cd /d "%~dp0"
if not exist test_logs mkdir test_logs
set "LOG=test_logs\setup_env.txt"
set "ACT=%USERPROFILE%\anaconda3\Scripts\activate.bat"
if not exist "%ACT%" goto noconda

call "%ACT%" base
echo ==== setup start %date% %time% ==== > "%LOG%"

call conda env list | findstr /B /C:"DQN " >nul
if not errorlevel 1 goto envexists
echo [1/4] Creating conda env DQN with Python 3.11 ...
call conda create -n DQN python=3.11 -y --override-channels -c conda-forge >> "%LOG%" 2>&1
if errorlevel 1 goto fail
goto activate
:envexists
echo [1/4] Env DQN already exists - reusing it.

:activate
call "%ACT%" DQN
python -c "import sys; assert sys.prefix.lower().endswith('dqn'), sys.prefix; print('env:', sys.executable)" >> "%LOG%" 2>&1
if errorlevel 1 goto fail

echo [2/4] Installing PyTorch CUDA 12.8 build - about 3 GB, may take a few minutes ...
python -m pip install --upgrade pip >> "%LOG%" 2>&1
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 >> "%LOG%" 2>&1
if errorlevel 1 goto fail

echo [3/4] Installing the rest of requirements.txt ...
python -m pip install -r requirements.txt >> "%LOG%" 2>&1
if errorlevel 1 goto fail

echo [4/4] Checking packages and running selftest ...
set KMP_DUPLICATE_LIB_OK=TRUE
echo ==== check_env ==== >> "%LOG%"
python test_logs\check_env.py >> "%LOG%" 2>&1
echo ==== selftest ==== >> "%LOG%"
python selftest.py >> "%LOG%" 2>&1
if errorlevel 1 goto fail

echo ==== END OK ==== >> "%LOG%"
echo.
echo Done. Env DQN is ready. Log: %LOG%
pause
exit /b 0

:noconda
echo Cannot find Anaconda at %USERPROFILE%\anaconda3
pause
exit /b 1

:fail
echo ==== FAILED ==== >> "%LOG%"
echo.
echo Something failed. See %LOG%
pause
exit /b 1
