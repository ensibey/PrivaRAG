@echo off
echo ================================================
echo   Local RAG System - Windows Setup
echo ================================================
echo.

echo [1/4] Creating Python virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.10+ first.
    pause
    exit /b 1
)
echo Done.

echo.
echo [2/4] Installing Python dependencies...
call venv\Scripts\activate
venv\Scripts\python.exe -m pip install --upgrade pip --quiet
venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: pip install failed. Check requirements.txt.
    pause
    exit /b 1
)
echo Done.

echo.
echo [3/4] Creating data directories...
if not exist "chroma_db" mkdir chroma_db
if not exist "uploaded_docs" mkdir uploaded_docs
echo Done.

echo.
echo [4/4] Checking Ollama models...
echo Pulling llama3 (this may take a while on first run)...
ollama pull llama3
echo Pulling nomic-embed-text...
ollama pull nomic-embed-text
echo Done.

echo.
echo ================================================
echo   Setup Complete!
echo ================================================
echo.
echo To start the app, run:
echo   venv\Scripts\activate
echo   streamlit run app.py
echo.
pause
