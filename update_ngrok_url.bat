@echo off
setlocal enabledelayedexpansion

REM Update ngrok URL Script (Windows Batch Version)
REM This script updates all crucial instances of ngrok URLs in the project
REM Usage: update_ngrok_url.bat <new_ngrok_url>
REM Example: update_ngrok_url.bat https://abc123-new-url.ngrok-free.app

echo 🔄 ngrok URL Update Script (Windows)
echo ====================================

REM Check if new URL is provided
if "%~1"=="" (
    echo ❌ Error: Usage: %0 ^<new_ngrok_url^>
    echo Example: %0 https://abc123-new-url.ngrok-free.app
    exit /b 1
)

set "new_url=%~1"

REM Basic validation - check if it starts with https and contains ngrok-free.app
echo %new_url% | findstr /r "^https://.*\.ngrok-free\.app$" >nul
if errorlevel 1 (
    echo ❌ Error: Invalid ngrok URL format: %new_url%
    echo Expected format: https://xxxx-xxx.ngrok-free.app
    exit /b 1
)

echo ✅ New ngrok URL: %new_url%
echo.

REM Get current directory
set "project_root=%~dp0"
cd /d "%project_root%"

echo 📍 Project root: %project_root%
echo.

REM Create backup timestamp
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "timestamp=%dt:~0,8%_%dt:~8,6%"

set "total_files=0"
set "updated_files=0"

REM Function to update a file
call :update_file "test_ngrok_connection.py" "ngrok connection test script"
call :update_file "debug_connection.py" "debug connection script"
call :update_file "test_history.py" "history test script"
call :update_file "code\llm_agent_linux_package\README.md" "LLM agent documentation"
call :update_file "code\llm_agent_linux_package\quick_test.sh" "quick test script"
call :update_file "code\llm_agent_linux_package\package_info.txt" "package info file"

echo.
echo 📊 Update Summary
echo =================
echo 📄 Total files processed: !total_files!
echo ✅ Successfully updated: !updated_files!

echo.
echo ✅ ngrok URL update completed!
echo 🔗 New URL: %new_url%

echo.
echo 🧪 To test the new URL, run:
echo    python test_transcriber_agent.py %new_url% --test-only

pause
exit /b 0

REM Function to update a file
:update_file
set /a total_files+=1
set "file=%~1"
set "description=%~2"

if not exist "%file%" (
    echo ⚠️  Warning: File not found: %file%
    goto :eof
)

REM Check if file contains ngrok URLs
findstr /c:"ngrok-free.app" "%file%" >nul
if errorlevel 1 (
    echo ℹ️  No ngrok URLs found in: %file%
    goto :eof
)

echo 🔄 Updating %description%: %file%

REM Create backup
if exist "%file%" (
    copy "%file%" "%file%.backup.%timestamp%" >nul
    echo 💾 Backed up: %file%
)

REM Use PowerShell to do the regex replacement (more reliable than batch)
powershell -Command "(Get-Content '%file%') | ForEach-Object { $_ -replace 'https://[a-zA-Z0-9-]*\.ngrok-free\.app', '%new_url%' } | Set-Content '%file%'"

if errorlevel 0 (
    set /a updated_files+=1
    echo ✅ Updated: %file%
) else (
    echo ❌ Failed to update: %file%
)

goto :eof
