@echo off
TITLE Enterprise Knowledge Assistant - Launchpad
echo ======================================================================
echo    Starting Enterprise Knowledge Assistant (Fullstack RAG)
echo ======================================================================
echo.

:: Detect Python
set PYTHON_CMD=python
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" (
        set "PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    ) else (
        echo [ERROR] Python not found in PATH or in AppData. Please verify your Python installation.
        pause
        exit /b 1
    )
)

echo [1/3] Using Python: %PYTHON_CMD%

:: Check if vector index exists
if not exist "faiss_index" (
    echo [INFO] FAISS vector index not found. Building index now...
    "%PYTHON_CMD%" src\knowledge_assistant\Ingestion\vector_store.py
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to build vector database. Check logs above.
        pause
        exit /b 1
    )
) else (
    echo [2/3] FAISS Vector Database found.
)

:: Start FastAPI Backend in a new window
echo [3/3] Starting Backend & Frontend...
start "Enterprise RAG - Backend (FastAPI :8000)" cmd /k ""%PYTHON_CMD%" -m uvicorn src.knowledge_assistant.api:app --host 127.0.0.1 --port 8000 --reload"

:: Start React Frontend in a new window
start "Enterprise RAG - Frontend (React :5173)" cmd /k "cd frontend && npm run dev"

echo.
echo ======================================================================
echo  Both services launched successfully!
echo.
echo  - Frontend Web UI: http://localhost:5173
echo  - Backend API:     http://127.0.0.1:8000
echo  - API Swagger Docs: http://127.0.0.1:8000/docs
echo ======================================================================
echo.
pause

