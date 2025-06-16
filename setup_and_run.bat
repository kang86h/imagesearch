@echo off
CHCP 65001 >nul
ECHO.
ECHO ==========================================================
ECHO   CLIP Image Search Project (CPU/GPU Auto-Detect)
ECHO ==========================================================
ECHO.

REM --- 1. 가상 환경(venv) 확인 및 생성 ---
IF NOT EXIST "venv\\Scripts\\activate.bat" (
    ECHO [1/5] Creating a new virtual environment...
    python -m venv venv
    IF ERRORLEVEL 1 (
        ECHO. & ECHO [ERROR] Failed to create a virtual environment.
        ECHO Please check if Python is installed and added to the PATH.
        pause & exit /b
    )
) ELSE (
    ECHO [1/5] Using existing virtual environment.
)

REM --- 2. 가상 환경 활성화 ---
ECHO [2/5] Activating virtual environment...
CALL "venv\\Scripts\\activate.bat"

REM --- 3. 공통 라이브러리 설치 ---
ECHO [3/5] Installing common libraries from requirements.txt...
pip install -r requirements.txt
IF ERRORLEVEL 1 (
    ECHO. & ECHO [ERROR] Failed to install common libraries.
    ECHO Please check requirements.txt and your internet connection.
    pause & exit /b
)

REM --- 4. PyTorch 설치 (GPU/CPU 자동 감지) ---
ECHO [4/5] Installing PyTorch...
where /q nvidia-smi
IF %ERRORLEVEL% EQU 0 (
    ECHO  - NVIDIA GPU detected. Installing PyTorch for GPU.
    pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
) ELSE (
    ECHO  - No NVIDIA GPU detected. Installing PyTorch for CPU.
    pip install torch torchvision torchaudio
)

IF ERRORLEVEL 1 (
    ECHO. & ECHO [ERROR] Failed to install PyTorch.
    ECHO Please check your internet connection and try again.
    pause & exit /b
)

REM --- 5. 파이썬 스크립트 실행 ---
ECHO [5/5] Running the image search application...
ECHO.
python search_app.py

ECHO.
ECHO The program has finished. Press any key to close this window.
pause >nul
