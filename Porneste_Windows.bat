@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if not errorlevel 1 (
    py -3 "%~dp0porneste.py"
    goto :finish
)
where python >nul 2>nul
if not errorlevel 1 (
    python "%~dp0porneste.py"
    goto :finish
)
echo Python nu a fost gasit. Instaleaza Python 3.10 sau mai nou de la python.org.
echo La instalare bifeaza Add Python to PATH, apoi ruleaza din nou acest fisier.
:finish
echo.
echo Apasa orice tasta pentru a inchide fereastra.
pause >nul
