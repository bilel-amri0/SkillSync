# SkillSync GitHub Upload Helper Script
# This script helps you prepare and upload your project to GitHub

Write-Host "🚀 SkillSync - GitHub Upload Helper" -ForegroundColor Cyan
Write-Host "====================================`n" -ForegroundColor Cyan

# Step 1: Check for large files
Write-Host "📊 Step 1: Checking for files larger than 50MB..." -ForegroundColor Yellow
$largeFiles = Get-ChildItem -Recurse -File -ErrorAction SilentlyContinue | 
    Where-Object { $_.Length -gt 50MB } | 
    Select-Object FullName, @{Name="SizeMB";Expression={[math]::Round($_.Length/1MB,2)}}

if ($largeFiles) {
    Write-Host "⚠️  WARNING: Found large files:" -ForegroundColor Red
    $largeFiles | Format-Table -AutoSize
    Write-Host "These files should be in .gitignore" -ForegroundColor Red
} else {
    Write-Host "✅ No files over 50MB found (good!)`n" -ForegroundColor Green
}

# Step 2: Check git status
Write-Host "📋 Step 2: Checking git status..." -ForegroundColor Yellow
if (Test-Path .git) {
    git status --short
} else {
    Write-Host "⚠️  Git not initialized. Run: git init" -ForegroundColor Red
}
Write-Host ""

# Step 3: Verify .gitignore
Write-Host "🔍 Step 3: Verifying .gitignore..." -ForegroundColor Yellow
if (Test-Path .gitignore) {
    Write-Host "✅ .gitignore exists" -ForegroundColor Green
    $gitignoreContent = Get-Content .gitignore -Raw
    
    $checks = @(
        @{Pattern = "\.venv"; Name = "Virtual environment"},
        @{Pattern = "\.env"; Name = "Environment files"},
        @{Pattern = "\*\.db"; Name = "Database files"},
        @{Pattern = "models/"; Name = "ML models"},
        @{Pattern = "node_modules/"; Name = "Node modules"}
    )
    
    foreach ($check in $checks) {
        if ($gitignoreContent -match $check.Pattern) {
            Write-Host "  ✅ $($check.Name) excluded" -ForegroundColor Green
        } else {
            Write-Host "  ⚠️  $($check.Name) NOT excluded" -ForegroundColor Red
        }
    }
} else {
    Write-Host "❌ .gitignore not found!" -ForegroundColor Red
}
Write-Host ""

# Step 4: Instructions
Write-Host "📝 Next Steps:" -ForegroundColor Cyan
Write-Host "===============`n" -ForegroundColor Cyan

if (-not (Test-Path .git)) {
    Write-Host "1. Initialize Git:" -ForegroundColor Yellow
    Write-Host "   git init`n" -ForegroundColor White
}

Write-Host "2. Add files to staging:" -ForegroundColor Yellow
Write-Host "   git add .`n" -ForegroundColor White

Write-Host "3. Create initial commit:" -ForegroundColor Yellow
Write-Host "   git commit -m `"Initial commit: SkillSync v2.1.0`"`n" -ForegroundColor White

Write-Host "4. Create GitHub repository:" -ForegroundColor Yellow
Write-Host "   - Go to: https://github.com/new" -ForegroundColor White
Write-Host "   - Name: SkillSync" -ForegroundColor White
Write-Host "   - Don't initialize with README`n" -ForegroundColor White

Write-Host "5. Connect and push:" -ForegroundColor Yellow
Write-Host "   git remote add origin https://github.com/YOUR_USERNAME/SkillSync.git" -ForegroundColor White
Write-Host "   git branch -M main" -ForegroundColor White
Write-Host "   git push -u origin main`n" -ForegroundColor White

Write-Host "📚 For detailed instructions, see: GITHUB_SETUP.md`n" -ForegroundColor Cyan

# Prompt to continue
Write-Host "Would you like to:" -ForegroundColor Yellow
Write-Host "  [1] Initialize git and add files" -ForegroundColor White
Write-Host "  [2] Just show git status" -ForegroundColor White
Write-Host "  [3] Exit" -ForegroundColor White
$choice = Read-Host "`nEnter choice (1-3)"

switch ($choice) {
    "1" {
        Write-Host "`n🔧 Initializing git..." -ForegroundColor Cyan
        if (-not (Test-Path .git)) {
            git init
        }
        Write-Host "📦 Adding files..." -ForegroundColor Cyan
        git add .
        Write-Host "`n✅ Files staged! Check with: git status" -ForegroundColor Green
        Write-Host "Next: git commit -m `"Initial commit: SkillSync v2.1.0`"" -ForegroundColor Yellow
    }
    "2" {
        if (Test-Path .git) {
            git status
        } else {
            Write-Host "Git not initialized" -ForegroundColor Red
        }
    }
    "3" {
        Write-Host "Exiting..." -ForegroundColor Gray
    }
}
