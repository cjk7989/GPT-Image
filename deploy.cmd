@echo off
setlocal

:: Load deploy config from .env
if not exist "%~dp0.env" (
    echo ERROR: .env file not found. Create it with AZURE_RG, AZURE_APP, AZURE_APP_URL.
    pause
    exit /b 1
)
for /f "usebackq tokens=1,* delims==" %%A in ("%~dp0.env") do (
    set "%%A=%%B"
)

set RG=%AZURE_RG%
set APP=%AZURE_APP%
set URL=%AZURE_APP_URL%

if "%RG%"=="" (
    echo ERROR: AZURE_RG not set in .env
    pause
    exit /b 1
)

echo ========================================
echo  GPT Image - Azure Deployment
echo ========================================

:: Auto-generate build version (timestamp)
for /f %%I in ('powershell -Command "Get-Date -Format yyyyMMddHHmmss"') do set "BUILD_VER=%%I"
echo Build version: %BUILD_VER%
echo %BUILD_VER%> "%~dp0src\version.txt"

:: Check Azure CLI login
call az account show >nul 2>&1
if %errorlevel% neq 0 (
    echo Logging in to Azure...
    call az login --use-device-code
)

:: Create zip and deploy
echo.
echo Deploying code...
pushd "%~dp0"

if exist deploy.zip del deploy.zip
echo Creating zip package...
powershell -ExecutionPolicy Bypass -Command "Compress-Archive -Path src\* -DestinationPath deploy.zip -Force"

if not exist deploy.zip (
    echo ERROR: Failed to create deploy.zip
    popd
    pause
    exit /b 1
)
echo Zip created. Uploading to Azure...

call az webapp deployment source config-zip --resource-group %RG% --name %APP% --src deploy.zip
set DEPLOY_ERR=%errorlevel%

if exist deploy.zip del deploy.zip
popd

if %DEPLOY_ERR% neq 0 (
    echo.
    echo ERROR: Deployment failed!
    pause
    exit /b 1
)

:: Re-set startup command (deployment may clear it)
echo.
echo Setting startup command...
call az webapp config set --resource-group %RG% --name %APP% --startup-file "gunicorn main:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --timeout 300" >nul

echo Restarting app...
call az webapp restart --resource-group %RG% --name %APP%

echo.
echo Done! %URL%
pause
