@echo off
CHCP 65001 >nul
ECHO.
ECHO ==========================================================
ECHO   CLIP 이미지 검색 프로젝트 (CPU/GPU 자동 감지)
ECHO ==========================================================
ECHO.

REM --- 1. 가상 환경(venv) 확인 및 생성 ---
IF NOT EXIST "venv\Scripts\activate.bat" (
    ECHO [1/5] 'venv' 폴더를 찾을 수 없습니다. 새로운 가상 환경을 생성합니다...
    python -m venv venv
    IF ERRORLEVEL 1 (
        ECHO. & ECHO [오류] 파이썬을 찾을 수 없거나 가상 환경 생성에 실패했습니다.
        ECHO 파이썬이 설치되어 있고 환경 변수(PATH)에 등록되었는지 확인하세요.
        pause & exit /b
    )
) ELSE (
    ECHO [1/5] 기존 가상 환경('venv')을 사용합니다.
)

REM --- 2. 가상 환경 활성화 ---
ECHO [2/5] 가상 환경을 활성화합니다...
CALL "venv\Scripts\activate.bat"

REM --- 3. 공통 라이브러리 설치 ---
ECHO [3/5] requirements.txt로 공통 라이브러리를 설치합니다...
pip install -r requirements.txt
IF ERRORLEVEL 1 (
    ECHO. & ECHO [오류] 공통 라이브러리 설치에 실패했습니다.
    ECHO requirements.txt 파일과 인터넷 연결을 확인하세요.
    pause & exit /b
)

REM --- 4. PyTorch 설치 (GPU/CPU 자동 감지) ---
ECHO [4/5] 시스템 환경을 분석하여 PyTorch를 설치합니다...

REM nvidia-smi.exe 존재 여부로 GPU 확인
where /q nvidia-smi
IF %ERRORLEVEL% EQU 0 (
    ECHO  - NVIDIA GPU가 감지되었습니다. GPU용 PyTorch를 설치합니다. (cu118)
    pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
) ELSE (
    ECHO  - NVIDIA GPU가 없습니다. CPU용 PyTorch를 설치합니다.
    pip install torch torchvision torchaudio
)

IF ERRORLEVEL 1 (
    ECHO. & ECHO [오류] PyTorch 설치에 실패했습니다.
    ECHO 인터넷 연결을 확인하거나, 잠시 후 다시 시도해주세요.
    pause & exit /b
)

REM --- 5. 파이썬 스크립트 실행 ---
ECHO [5/5] 이미지 검색 프로그램을 실행합니다...
ECHO.
python search_app.py

ECHO.
ECHO 프로그램이 종료되었습니다. 아무 키나 누르면 창이 닫힙니다.
pause >nul
