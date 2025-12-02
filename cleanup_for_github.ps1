# Script de nettoyage pour preparer le projet pour GitHub
# Usage: .\cleanup_for_github.ps1

Write-Host "`n=== NETTOYAGE DU PROJET POUR GITHUB ===`n" -ForegroundColor Cyan

$itemsToRemove = @()

# Fichiers de test temporaires
Write-Host "Recherche des fichiers de test temporaires..." -ForegroundColor Yellow
$testFiles = Get-ChildItem -Path . -Filter "test_*.py" -File | Where-Object { $_.Name -notlike "test_api.py" }
$checkFiles = Get-ChildItem -Path . -Filter "check_*.py" -File
$TESTFiles = Get-ChildItem -Path . -Filter "TEST_*.py" -File
$itemsToRemove += $testFiles + $checkFiles + $TESTFiles

$otherTestFiles = @(
    "testapi.py",
    "example_usage.py",
    "run.py",
    "setup.py"
)
foreach ($file in $otherTestFiles) {
    if (Test-Path $file) {
        $itemsToRemove += Get-Item $file
    }
}

# Fichiers de documentation redondants
Write-Host "Recherche des fichiers de documentation redondants..." -ForegroundColor Yellow
$docPatterns = @(
    "*_FIX.md",
    "*_GUIDE.md",
    "*_SUMMARY.md",
    "*_README.md",
    "RESUME_*.md",
    "PAGINATION_*.md",
    "TRENDS_*.md",
    "MIGRATION_*.md",
    "API_GUIDE.md",
    "ARCHITECTURE.md",
    "CLEANUP_REPORT.md",
    "CONTRIBUTING.md",
    "GITHUB_PUSH_CHECKLIST.md",
    "LLM_INTEGRATION.md",
    "QUICKSTART.md",
    "QUICKSTART_LLM.md",
    "README_COMPLET.md",
    "README_OLD.md",
    "database_schema.sql"
)

foreach ($pattern in $docPatterns) {
    $files = Get-ChildItem -Path . -Filter $pattern -File
    $itemsToRemove += $files
}

# Dossiers temporaires
Write-Host "Recherche des dossiers temporaires..." -ForegroundColor Yellow
$foldersToCheck = @(
    "data",
    "logs",
    ".pytest_cache"
)

foreach ($folder in $foldersToCheck) {
    if (Test-Path $folder) {
        $itemsToRemove += Get-Item $folder
    }
}

# Fichiers log
Write-Host "Recherche des fichiers log..." -ForegroundColor Yellow
$logFiles = Get-ChildItem -Path . -Filter "*.log" -File -Recurse -ErrorAction SilentlyContinue
$itemsToRemove += $logFiles

# Fichier .env (securite)
Write-Host "Verification du fichier .env..." -ForegroundColor Yellow
if (Test-Path ".env") {
    Write-Host "   ATTENTION: Fichier .env trouve!" -ForegroundColor Red
    Write-Host "   Ce fichier contient vos cles API et NE DOIT PAS etre pushe sur GitHub!" -ForegroundColor Red
    $confirmation = Read-Host "   Voulez-vous le supprimer? (o/n)"
    if ($confirmation -eq "o") {
        $itemsToRemove += Get-Item ".env"
    }
}

# Afficher le resume
Write-Host "`n=== RESUME ===" -ForegroundColor Cyan
Write-Host "   Fichiers a supprimer: $($itemsToRemove.Count)" -ForegroundColor White

if ($itemsToRemove.Count -eq 0) {
    Write-Host "`nAucun fichier a nettoyer! Le projet est propre.`n" -ForegroundColor Green
    exit 0
}

Write-Host "`nListe des elements a supprimer:" -ForegroundColor Yellow
foreach ($item in $itemsToRemove) {
    if ($item.PSIsContainer) {
        Write-Host "   [DOSSIER] $($item.Name)" -ForegroundColor Magenta
    } else {
        Write-Host "   [FICHIER] $($item.Name)" -ForegroundColor Gray
    }
}

# Demander confirmation
Write-Host "`nCette operation est IRREVERSIBLE!" -ForegroundColor Red
$confirmation = Read-Host "Voulez-vous continuer? (o/n)"

if ($confirmation -eq "o") {
    Write-Host "`nSuppression en cours..." -ForegroundColor Yellow
    
    $deleted = 0
    $errors = 0
    
    foreach ($item in $itemsToRemove) {
        try {
            if ($item.PSIsContainer) {
                Remove-Item -Path $item.FullName -Recurse -Force -ErrorAction Stop
            } else {
                Remove-Item -Path $item.FullName -Force -ErrorAction Stop
            }
            $deleted++
            Write-Host "   [OK] $($item.Name)" -ForegroundColor Green
        } catch {
            $errors++
            Write-Host "   [ERREUR] $($item.Name) - $($_.Exception.Message)" -ForegroundColor Red
        }
    }
    
    Write-Host "`n=== NETTOYAGE TERMINE ===" -ForegroundColor Green
    Write-Host "   Supprimes: $deleted" -ForegroundColor White
    Write-Host "   Erreurs: $errors" -ForegroundColor White
    
    # Verifier .gitignore
    if (Test-Path ".gitignore") {
        Write-Host "`n[OK] .gitignore est present" -ForegroundColor Green
    } else {
        Write-Host "`n[ATTENTION] .gitignore manquant!" -ForegroundColor Red
    }
    
    # Verifier README
    if (Test-Path "README.md") {
        Write-Host "[OK] README.md est present" -ForegroundColor Green
    } else {
        Write-Host "[ATTENTION] README.md manquant!" -ForegroundColor Red
    }
    
    Write-Host "`n=== PROCHAINES ETAPES ===" -ForegroundColor Cyan
    Write-Host "   1. Verifier les fichiers restants: Get-ChildItem" -ForegroundColor White
    Write-Host "   2. Initialiser Git: git init" -ForegroundColor White
    Write-Host "   3. Ajouter les fichiers: git add ." -ForegroundColor White
    Write-Host "   4. Commit: git commit -m `"Initial commit`"" -ForegroundColor White
    Write-Host "   5. Creer un repo sur github.com" -ForegroundColor White
    Write-Host "   6. Lier le repo: git remote add origin https://github.com/votre-username/projet_deep_learning.git" -ForegroundColor White
    Write-Host "   7. Push: git push -u origin main`n" -ForegroundColor White
    
} else {
    Write-Host "`nNettoyage annule.`n" -ForegroundColor Yellow
}
