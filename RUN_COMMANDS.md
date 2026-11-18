# How to Run the Project

### 1. Start Backend (Terminal 1)
```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe" src\knowledge_assistant\api.py
```
*Backend runs at: `http://127.0.0.1:8000`*

---

### 2. Start Frontend (Terminal 2)
```powershell
cd frontend
npm run dev
```
*Open in your browser: **`http://localhost:5173`***

---

### (Or use 1-Click Launch)
Double-click:
```cmd
run_project.bat
```
*(Starts both services automatically in separate windows).*
