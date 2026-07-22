# Start both API and Streamlit for Real Estate Valuation App
# Usage: .\run.ps1

Write-Host "🚀 Starting Real Estate Valuation App..." -ForegroundColor Green
Write-Host ""

# Check Python
python --version | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Python not found" -ForegroundColor Red
    exit 1
}

# Start API in new PowerShell window
Write-Host "📡 Starting API on port 8000..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"

# Wait for API to start
Start-Sleep -Seconds 3

# Start Streamlit in new PowerShell window
Write-Host "🎨 Starting Streamlit on port 8500..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; streamlit run app/ui/streamlit_app.py --server.port 8500"

Write-Host ""
Write-Host "✅ Both services started!" -ForegroundColor Green
Write-Host ""
Write-Host "📡 API:       http://localhost:8000/docs" -ForegroundColor Yellow
Write-Host "🎨 Streamlit: http://localhost:8500" -ForegroundColor Yellow
Write-Host ""
Write-Host "Close the PowerShell windows to stop the services." -ForegroundColor Gray
