@echo off
cd /d "%~dp0"
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" DQN
set KMP_DUPLICATE_LIB_OK=TRUE
echo [1/2] selftest (fake game, ~1 min) ...
python selftest.py > test_logs\selftest.txt 2>&1
findstr /C:"PASS - pipeline works" test_logs\selftest.txt >nul || (
  echo selftest FAILED - see test_logs\selftest.txt
  pause
  exit /b 1
)
echo selftest passed.
echo.
echo [2/2] 3000-step pilot. Click the Dead Cells window now (same as last time).
echo       Stop early: hold q, do NOT close this window (otherwise nothing is saved).
python train_ablation.py --run-dir runs_pilot\pilot3000 --max-steps 3000
echo.
echo ==== pilot finished ====
pause
