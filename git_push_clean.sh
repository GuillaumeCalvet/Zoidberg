#!/bin/bash

set -e

echo "📁 Passage dans le dossier ~/Zoidberg"
cd ~/Zoidberg || { echo "❌ Dossier introuvable"; exit 1; }

echo "🧹 Nettoyage ancien dépôt Git s'il existe"
rm -rf .git

echo "🚀 Initialisation Git"
git init

echo "⚙️ Configuration nom de branche par défaut : main"
git branch -M main

echo "👤 Configuration de l'identité Git"
git config user.name "GuiCa30"
git config user.email "guillaume.calvet@gmail.com"

echo "📦 Installation Git LFS si besoin"
if ! command -v git-lfs &> /dev/null; then
    sudo apt update && sudo apt install git-lfs -y
fi

git lfs install

echo "📂 Suivi des gros fichiers avec Git LFS"
git lfs track "*.pt" "*.pth" "*.h5" "*.zip" "*.tar" "*.gz" "*.npz" "*.pkl"
git lfs track "models/**"
git lfs track "data/**"
git lfs track "uploads/**"
git add .gitattributes

echo "📝 Génération du fichier .gitignore"
cat > .gitignore <<EOF
# Fichiers à ignorer
__pycache__/
.ipynb_checkpoints/
*.pyc
*.log

# Dossiers lourds gérés par LFS
models/
uploads/
data/

# Fichiers d'environnement
.env
EOF

git add .gitignore

echo "✅ Ajout des fichiers légers"
for item in mobile notebooks Web docs Zoidberg tests requirements.txt setup.cfg pyproject.toml; do
  if [ -e "$item" ]; then
    git add "$item"
  else
    echo "ℹ️ Skipped: $item (inexistant)"
  fi
done

echo "🧠 Commit des fichiers légers"
git commit -m "Ajout des composants principaux (code, docs, notebooks, web)" || echo "⚠️ Aucun fichier léger à commit"

echo "📦 Ajout des fichiers lourds (via LFS, forcé car ignorés)"
for item in models data uploads; do
  if [ -d "$item" ]; then
    git add -f "$item"
  else
    echo "ℹ️ Skipped: $item (dossier manquant)"
  fi
done

echo "🧠 Commit des fichiers lourds"
git commit -m "Ajout des modèles, datasets et uploads avec Git LFS" || echo "⚠️ Aucun fichier lourd à commit"

read -p "🔗 URL du dépôt GitHub (SSH recommandé) : " GIT_URL
git remote add origin "$GIT_URL"

echo "🚀 Push vers GitHub"
git push -u origin main

echo "✅ Projet Zoidberg envoyé avec succès !"
