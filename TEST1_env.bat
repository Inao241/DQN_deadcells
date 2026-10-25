@echo off
cd /d "%~dp0"
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" DQN
echo ==== where python ==== > test_logs\test1.txt
where python >> test_logs\test1.txt 2>&1
set KMP_DUPLICATE_LIB_OK=TRUE
python test_logs\check_env.py >> test_logs\test1.txt 2>&1
echo ==== selftest ==== >> test_logs\test1.txt
python selftest.py >> test_logs\test1.txt 2>&1
echo ==== END ==== >> test_logs\test1.txt
