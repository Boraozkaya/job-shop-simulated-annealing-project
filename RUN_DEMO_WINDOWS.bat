@echo off
cd /d "%~dp0"
echo Job Shop Scheduling with Simulated Annealing
echo Running live demo...
echo.
py -3 demo_run.py --instance ft10 --iterations 2500
if errorlevel 1 (
  python demo_run.py --instance ft10 --iterations 2500
)
echo.
pause
