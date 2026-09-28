@echo off
chcp 65001 >nul
set "PROJECT_ROOT=%~dp0"
echo [SOCI-AI] 正在调用 PowerShell 启动本地实时自检……
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%PROJECT_ROOT%start_showcase.ps1"
if errorlevel 1 (
  echo [SOCI-AI] 启动失败，请查看上方提示。
  pause
)
