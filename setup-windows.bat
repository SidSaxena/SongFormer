@echo off
setlocal

echo === SongFormer Setup (Windows + CUDA GPU) ===
echo.
echo Prerequisites:
echo   - Conda/Miniconda installed
echo   - NVIDIA GPU with updated drivers
echo   - For Docker: Docker Desktop with WSL2 backend
echo.

:: Check if conda is available
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: conda is not in PATH.
    echo.
    echo Fix: Run this script from an Anaconda Prompt, or initialize conda:
    echo   1. Open Anaconda Prompt from the Start Menu
    echo   2. Navigate to this directory: cd %cd%
    echo   3. Run: setup-windows.bat
    echo.
    echo Alternatively, run 'conda init cmd.exe' once, then retry.
    exit /b 1
)

:: Create conda environment
call conda create -n songformer python=3.10 -y
if %errorlevel% neq 0 (
    echo ERROR: Failed to create conda environment.
    exit /b 1
)
call conda activate songformer

:: Install PyTorch with CUDA support
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

:: Install remaining dependencies (excludes pesq and gpustat — no C compiler needed)
pip install -r requirements-no-build.txt

:: Download model checkpoints
python -c "import os; os.chdir('src/SongFormer'); from utils.fetch_pretrained import download_all; download_all()"

echo.
echo === Setup complete! ===
echo Run:  conda activate songformer
echo Then: python app.py
echo.
echo For a public share link: set SONGFORMER_SHARE=1 ^& python app.py

endlocal
