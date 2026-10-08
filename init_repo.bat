@echo off
set remote_url=%~1

if "%remote_url%"=="" (
    echo Error: Please provide your Git remote URL.
    echo Usage: .\init_repo.bat https://github.com/your-username/your-repo.git
    exit /b 1
)

echo Initializing git repository...
git init

echo Adding files...
git add .

echo Committing...
git commit -m "Initial commit"

echo Setting branch to main...
git branch -M main

echo Adding remote repository...
git remote add origin %remote_url%

echo Pushing to remote...
git push -u origin main
