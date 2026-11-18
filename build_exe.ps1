$ErrorActionPreference = "Stop"

Write-Host "Building AI Demand Forecaster executable..."

if (-not (Test-Path ".\.venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
}

.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

$launcher = @'
from streamlit.web import cli as stcli
import os
import sys


def resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


if __name__ == "__main__":
    app_path = resource_path("app.py")
    sys.argv = ["streamlit", "run", app_path, "--server.headless=false", "--global.developmentMode=false"]
    sys.exit(stcli.main())
'@

Set-Content -Path ".\streamlit_launcher.py" -Value $launcher -Encoding UTF8

.\.venv\Scripts\pyinstaller.exe `
    --noconfirm `
    --clean `
    --name "AI_Demand_Forecaster" `
    --add-data "app.py;." `
    --add-data "demand_forecaster;demand_forecaster" `
    --add-data "data;data" `
    --hidden-import "sklearn.ensemble._gb" `
    --hidden-import "xgboost" `
    --hidden-import "statsmodels.tsa.arima.model" `
    streamlit_launcher.py

Write-Host "Build complete. Open dist\AI_Demand_Forecaster\AI_Demand_Forecaster.exe"
