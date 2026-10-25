@echo off
cd /d "%~dp0"
set KMP_DUPLICATE_LIB_OK=TRUE
echo ==== test ==== > test_logs\test2.txt
python test_logs\test_speed.py >> test_logs\test2.txt 2>&1
echo ==== END ==== >> test_logs\test2.txt
