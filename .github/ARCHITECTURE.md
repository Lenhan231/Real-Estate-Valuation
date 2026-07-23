# 📁 Repository Structure & File Reference

Complete guide to all files and directories in the repository root.

---

## 🗂️ Root Directory Overview

```
Real-Estate-Valuation/
│
├── 📄 Documentation (Start here!)
│   ├── README.md                 ← Project overview + quick start
│   ├── DEPLOYMENT.md             ← Deploy to any platform
│   └── .github/ARCHITECTURE.md   ← This file
│
├── 🔧 Configuration Files
│   ├── render.yaml               ← Render deployment config (2 services)
│   ├── Dockerfile                ← API container image
│   ├── pytest.ini                ← Python testing config
│   ├── .gitignore                ← Git ignore patterns
│   └── .gitattributes            ← Git line endings config
│
├── 📦 Dependencies
│   ├── requirements.txt           ← Python packages (pip)
│   ├── requirements-dev.txt       ← Dev/testing packages
│   ├── package.json               ← Node packages (legacy, unused)
│   └── package-lock.json          ← Node lock file (legacy, unused)
│
├── 🚀 Quick Start Scripts
│   ├── run.sh                     ← Linux/Mac: start API + Streamlit
│   └── run.ps1                    ← Windows: start API + Streamlit
│
├── 🏗️ Application Code (Documented)
│   ├── app/                       → FastAPI backend + Streamlit UI
│   ├── pipeline/                  → ETL orchestration
│   ├── models/                    → ML models & training
│   ├── data/                      → Data layers (raw, processed, etc.)
│   ├── notebooks/                 → Analysis & experiments
│   ├── tests/                     → Unit tests
│   └── scripts/                   → Utility scripts
│
├── 📚 Additional Config
│   ├── .deployment/               → Docker & deployment configs
│   ├── .github/                   → GitHub workflows & templates
│   ├── docs/                      → Additional documentation
│   └── .claude/                   → Claude Code config
│
└── 🔐 Environment (Not in git)
    ├── .env                       ← Local secrets (NEVER commit!)
    ├── .env.example               ← Template for .env
    └── .venv/                     ← Python virtual environment
```

---

## 📄 Root-Level Files Explained

### Documentation

| File | Purpose | Read When |
|------|---------|-----------|
| **README.md** | Main project overview, quick start, architecture | First time setup |
| **DEPLOYMENT.md** | Complete deployment guide for all platforms | Deploying to production |
| **.github/ARCHITECTURE.md** | This file — repository structure reference | Understanding repo layout |

---

### 🔧 Configuration Files

#### **render.yaml**

Render platform deployment configuration.

**Contains:**
- 2 services: `real-estate-valuation-api` (FastAPI) + `real-estate-valuation-streamlit` (Streamlit)
- Docker image paths
- Environment variables
- Auto-deploy settings

**When to edit:**
- Changing deployment settings
- Adding new services
- Updating environment variables

**Related:** [DEPLOYMENT.md](../DEPLOYMENT.md)

---

#### **Dockerfile**

Production Docker image for FastAPI API.

**Features:**
- Multi-stage build (small final image)
- Non-root user (security)
- Health checks
- Dynamic port support

**When to edit:**
- Adding system dependencies
- Changing base image
- Updating startup command

**Related:** [.deployment/README.md](../.deployment/README.md)

---

#### **pytest.ini**

Python testing framework configuration.

**Configures:**
```ini
[pytest]
testpaths = tests/        # Where to find tests
python_files = test_*.py  # Test file naming pattern
```

**When to edit:**
- Adding test directories
- Changing test discovery patterns

**Related:** `tests/` directory

---

#### **.gitignore**

Git ignore patterns (keep sensitive files out of repo).

**Ignores:**
- `.venv/` - Virtual environment
- `__pycache__/` - Python cache
- `.env` - Local secrets
- `*.pyc` - Compiled Python
- `.pytest_cache/` - Test cache
- `wandb/` - W&B artifacts

**When to edit:**
- Adding new build artifacts
- Adding new temporary files
- Adding new secret file patterns

**⚠️ CRITICAL:** Never remove `.env` from .gitignore!

---

#### **.gitattributes**

Git line-ending configuration (Windows ↔ Unix consistency).

**Configures:**
```
* text=auto              # Auto-detect text files
*.py text eol=lf         # Python always LF
*.ps1 text eol=crlf      # PowerShell always CRLF
*.jpg binary             # Images are binary
```

**When to edit:**
- Adding files with specific line-ending requirements
- Rare — usually not needed

---

### 📦 Dependencies

#### **requirements.txt**

Python package dependencies for production.

**Contains:**
```
streamlit>=1.57
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
lightgbm>=4.0
xgboost>=2.0
catboost>=1.2
fastapi>=0.100
uvicorn>=0.23
pydantic>=2.0
requests>=2.31
supabase>=2.31
# ... 20+ more
```

**When to edit:**
- Adding new Python packages
- Upgrading package versions
- Removing unused packages

**Install:**
```bash
pip install -r requirements.txt
```

---

#### **requirements-dev.txt**

Development & testing dependencies (not in production).

**Contains:**
```
pytest>=7.0
pytest-cov>=4.0
black>=23.0
flake8>=6.0
jupyter>=1.0
# ... testing & dev tools
```

**When to edit:**
- Adding dev tools (linters, formatters, notebooks)
- Adding testing libraries

**Install:**
```bash
pip install -r requirements-dev.txt
```

---

#### **package.json** & **package-lock.json**

Node.js configuration (legacy, can be removed).

**Status:** ⚠️ **UNUSED** — kept for reference only

**Can be safely deleted** if no Node.js tooling needed.

---

### 🚀 Quick Start Scripts

#### **run.sh** (Linux/Mac)

Start both API and Streamlit with one command.

**Does:**
```bash
# Start API (background)
python -m uvicorn app.main:app --port 8000 &

# Start Streamlit
streamlit run app/ui/streamlit_app.py --server.port 8501
```

**Usage:**
```bash
bash run.sh
```

**Alternative:** Use `docker-compose up` for containerized setup

---

#### **run.ps1** (Windows)

PowerShell version of run.sh for Windows.

**Does:** Same as run.sh but using PowerShell syntax

**Usage:**
```powershell
.\run.ps1
```

**Alternative:** Use `docker-compose up` for containerized setup

---

## 🏗️ Application Directories (Documented)

### Core Application

| Directory | Purpose | See |
|-----------|---------|-----|
| **app/** | FastAPI backend + Streamlit UI | [app/README.md](../app/README.md) |
| **pipeline/** | ETL data processing | [pipeline/README.md](../pipeline/README.md) |
| **models/** | ML models & training | [MODELS.md](../MODELS.md) |

### Data & Analysis

| Directory | Purpose | See |
|-----------|---------|-----|
| **data/** | Raw, processed, external, cache | [data/README.md](../data/README.md) |
| **notebooks/** | EDA, training, XAI analysis | [notebooks/README.md](../notebooks/README.md) |

### Testing & Utilities

| Directory | Purpose | See |
|-----------|---------|-----|
| **tests/** | Unit tests | pytest.ini |
| **scripts/** | Utility scripts | Individual files |
| **docs/** | Additional documentation | Various |

---

## 🔐 Environment Files (NOT in Git)

### **.env** (Local Secrets)

**⚠️ CRITICAL: NEVER COMMIT THIS FILE**

**Contains:**
```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=sb_secret_xxxxx
WANDB_API_KEY=your_api_key
API_URL=http://localhost:8000
```

**Create from template:**
```bash
cp .env.example .env
# Edit .env with your actual credentials
```

---

### **.env.example**

Template for `.env` file (safe to commit).

**Shows structure** without revealing secrets.

**When to edit:**
- Adding new environment variable
- Changing variable names

---

### **.venv/** (Virtual Environment)

Python virtual environment (NOT in git).

**Created by:**
```bash
python -m venv .venv
.venv/Scripts/activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

**In .gitignore:** ✓ Yes, correctly ignored

---

## 📊 Hidden Directories

### **.github/**

GitHub-specific files (workflows, templates, config).

**Contains:**
- `.github/workflows/` - CI/CD pipelines
- `.github/ARCHITECTURE.md` - This file
- PR/Issue templates

---

### **.deployment/**

Docker and deployment configuration.

**Contains:**
- `Dockerfile` - Production image
- `Dockerfile.streamlit` - Streamlit image
- `docker-compose.yml` - Local dev setup
- `start.sh` - Container entrypoint
- `run.py` - Streamlit launcher

**Related:** [.deployment/README.md](../.deployment/README.md)

---

### **.claude/**

Claude Code configuration.

**Contains:**
- `settings.json` - VS Code settings
- Project-specific configuration

---

### **wandb/** (Artifact Cache)

Weights & Biases experiment tracking (auto-generated).

**In .gitignore:** ✓ Yes, correctly ignored

---

## 🧹 File Cleanup Guide

### Safe to Remove

These can be safely deleted:

```bash
# Unused Node.js files
rm package.json package-lock.json

# Auto-generated caches (regenerated on next run)
rm -rf .pytest_cache wandb/
```

### Never Delete

These are critical:

```
.gitignore          # Prevents accidental commits
.env.example        # Template for secrets
requirements.txt    # Package dependencies
render.yaml         # Deployment config
Dockerfile          # Container definition
```

### Frequently Edited

| File | When | How Often |
|------|------|-----------|
| `.env` | Add new API keys/credentials | Per developer |
| `requirements.txt` | Add new Python packages | Per feature |
| `render.yaml` | Change deployment config | Rarely |
| `Dockerfile` | Add system dependencies | Rarely |

---

## 🚀 Common Workflows

### Local Development Setup

```bash
# 1. Clone repo
git clone https://github.com/Lenhan231/Real-Estate-Valuation.git
cd Real-Estate-Valuation

# 2. Setup environment
cp .env.example .env
# Edit .env with your credentials

# 3. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# 4. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 5. Start development
bash run.sh  # or .\run.ps1 on Windows
```

### Add New Python Package

```bash
# 1. Install package locally
pip install package-name

# 2. Update requirements
pip freeze | grep package-name >> requirements.txt

# 3. Commit changes
git add requirements.txt
git commit -m "add: package-name for [reason]"
```

### Deploy to Render

```bash
# 1. Changes are auto-deployed on git push
git push origin main

# 2. Render detects changes via render.yaml
# 3. Services rebuild and restart automatically

# 4. Check status
# Visit: https://dashboard.render.com
```

---

## 📚 Related Documentation

- [README.md](../README.md) - Project overview
- [DEPLOYMENT.md](../DEPLOYMENT.md) - Deployment guide
- [app/README.md](../app/README.md) - Backend/frontend
- [pipeline/README.md](../pipeline/README.md) - ETL process
- [data/README.md](../data/README.md) - Data layers
- [notebooks/README.md](../notebooks/README.md) - Analysis
- [.deployment/README.md](../.deployment/README.md) - Docker config

---

## 🆘 Quick Troubleshooting

| Issue | Check | Fix |
|-------|-------|-----|
| "Module not found" | requirements.txt | `pip install -r requirements.txt` |
| "Permission denied" run.sh | File permissions | `chmod +x run.sh` |
| "Port already in use" | Other running services | Kill process: `lsof -i :8000` |
| ".env not loading" | .env exists & readable | Verify path, check syntax |
| "Docker build fails" | Dockerfile syntax | Check [.deployment/README.md](../.deployment/README.md) |

---

**Last Updated:** 2026-07-23
**Repository:** Real-Estate-Valuation (DSP391m Capstone)
