@echo off
set message=%~1
if "%message%"=="" set message="Update"
git add .
git commit -m "%message%"
git push
