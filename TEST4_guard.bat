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
echo [2/2] 300-step real-game check of the preflight / RunGuard. Click the Dead Cells window now.
echo       Expect a line "preflight OK: player HP 15482 / 15482, boss HP ..." after the first reset.
if exist runs_pilot\guard300 ren runs_pilot\guard300 guard300_old_%RANDOM%
python train_ablation.py --run-dir runs_pilot\guard300 --max-steps 300 --expect-player-hp 15482
echo.
echo ==== exit code %ERRORLEVEL% (0 = OK, 3 = RunGuard / game error) ====
pause
