<#
.SYNOPSIS
    Launches the Enterprise Knowledge Assistant (Backend + Frontend) in separate terminals.
.DESCRIPTION
    Detects Python, verifies FAISS index, and launches FastAPI (:8000) and React (:5173).
#>

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "    Starting Enterprise Knowledge Assistant (Fullstack RAG)" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Resolve Python Path
$PythonPath = "python"
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    $DefaultPy310 = "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe"
    if (Test-Path $DefaultPy310) {
        $PythonPath = $DefaultPy310
    } else {
        Write-Host "[ERROR] Python was not found in PATH or at $DefaultPy310" -ForegroundColor Red
        Exit 1
    }
}
Write-Host "[1/3] Using Python: $PythonPath" -ForegroundColor Green

# 2. Check if FAISS index exists
if (-not (Test-Path "faiss_index")) {
    Write-Host "[INFO] FAISS index missing. Generating embeddings & building index..." -ForegroundColor Yellow
    & $PythonPath src\knowledge_assistant\Ingestion\vector_store.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Ingestion failed. Check error above." -ForegroundColor Red
        Exit 1
    }
} else {
    Write-Host "[2/3] FAISS Vector Database found." -ForegroundColor Green
}

# 3. Launch Backend in new PowerShell window
Write-Host "[3/3] Launching FastAPI Backend on port 8000..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Write-Host '=== Enterprise RAG - Backend (FastAPI :8000) ===' -ForegroundColor Green; & '$PythonPath' -m uvicorn src.knowledge_assistant.api:app --host 127.0.0.1 --port 8000 --reload"

# 4. Launch Frontend in new PowerShell window
Write-Host "[3/3] Launching React Frontend on port 5173..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Write-Host '=== Enterprise RAG - Frontend (React :5173) ===' -ForegroundColor Cyan; cd frontend; npm run dev"

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Green
Write-Host "  Both services launched successfully!" -ForegroundColor Green
Write-Host "  - Frontend Web UI:  http://localhost:5173" -ForegroundColor White
Write-Host "  - Backend API:      http://127.0.0.1:8000" -ForegroundColor White
Write-Host "  - API Swagger Docs: http://127.0.0.1:8000/docs" -ForegroundColor White
Write-Host "======================================================================" -ForegroundColor Green

