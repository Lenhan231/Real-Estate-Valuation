# 🐳 Deployment Configuration

Docker & deployment files for containerized deployments across Render, DigitalOcean, AWS, and local development.

---

## 📁 Directory Structure

```
.deployment/
├── README.md                    ← This file
├── Dockerfile                   ← Production API (FastAPI)
├── Dockerfile.streamlit         ← Streamlit UI container
├── docker-compose.yml           ← Local development (both services)
├── .dockerignore                ← Exclude files from Docker build
├── start.sh                      ← Entrypoint for Dockerfile
└── run.py                        ← Python wrapper for Streamlit startup
```

---

## 📋 File Descriptions

### **Dockerfile** (Production API)

Multi-stage build for FastAPI backend.

**Features:**
- ✅ Slim Python 3.11 base image
- ✅ Multi-stage build (build → runtime)
- ✅ Non-root user (security)
- ✅ Health checks
- ✅ Dynamic port support

**Usage:**
```bash
# Build
docker build -f .deployment/Dockerfile -t api:latest .

# Run
docker run -p 8000:8000 -e PORT=8000 api:latest
```

**Exposed:** Port 8000

**Entry:** `python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT`

---

### **Dockerfile.streamlit** (Streamlit UI)

Docker build for Streamlit frontend.

**Features:**
- ✅ Same base as main Dockerfile
- ✅ Streamlit runtime
- ✅ Dynamic port support
- ✅ Health check via HTTP

**Usage:**
```bash
# Build
docker build -f .deployment/Dockerfile.streamlit -t streamlit:latest .

# Run
docker run -p 8501:8501 -e PORT=8501 \
  -e API_URL=http://api:8000 streamlit:latest
```

**Exposed:** Port 8501

**Entry:** `python run.py` (via start.sh)

---

### **docker-compose.yml** (Local Development)

Run both API + Streamlit locally with Docker Compose.

**Services:**
- `api` - FastAPI backend (port 8000)
- `streamlit` - Streamlit frontend (port 8501)

**Usage:**

```bash
# Start both services
docker-compose up

# Start API only
docker-compose up api

# Build images
docker-compose build

# View logs
docker-compose logs -f api
docker-compose logs -f streamlit

# Stop all
docker-compose down

# Full cleanup (remove volumes)
docker-compose down -v
```

**Environment:** Reads from `.env` file

**Networks:** Internal Docker network for API ↔ Streamlit communication

---

### **.dockerignore**

Files excluded from Docker build context (speeds up builds, reduces image size).

**Excludes:**
- `.git/` - Git history (not needed in container)
- `__pycache__/` - Python cache
- `.venv/` - Local virtual env
- `*.pyc` - Compiled Python
- `.env.local` - Local secrets (don't leak into image)
- `node_modules/` - If using Node
- `*.log` - Logs
- `.pytest_cache/` - Test cache

---

### **start.sh**

Shell entrypoint for Dockerfile containers.

**Does:**
1. Print debug info (shell, PATH, PORT, etc.)
2. Execute Python entrypoint

**Used By:** Both Dockerfile and Dockerfile.streamlit

**Debug Output:**
```bash
=== START DEBUG ===
Shell: /bin/sh
PATH: /usr/local/bin:/usr/local/sbin:...
PORT: 8501
Python location: /usr/local/bin/python3
=== END DEBUG ===
```

---

### **run.py**

Python wrapper for Streamlit startup (used by Dockerfile.streamlit).

**Features:**
- ✅ Debug logging (Python version, cwd, env vars)
- ✅ Catches startup errors with traceback
- ✅ Uses `os.execvp()` to replace process (proper signal handling)
- ✅ Runs: `python -m streamlit run app/ui/streamlit_app.py`

**Debug Output:**
```
[DEBUG] Python version: 3.11.15
[DEBUG] Working directory: /app
[DEBUG] PORT: 8501
[DEBUG] API_URL: https://api.example.com
[DEBUG] Streamlit imported successfully (v1.60.0)
[DEBUG] Running: python -m streamlit run ...
```

---

## 🚀 Deployment Workflows

### Local Development (Docker Compose)

```bash
# Clone repo
git clone ...
cd Real-Estate-Valuation

# Create .env file
cp .env.example .env
# Edit .env with your credentials

# Start with Docker Compose
docker-compose up

# Access:
# - API: http://localhost:8000/docs
# - Streamlit: http://localhost:8501
```

---

### Render Deployment (Automatic)

Uses `render.yaml` at project root.

**Services:**
1. `real-estate-valuation-api` - FastAPI
2. `real-estate-valuation-streamlit` - Streamlit

**Automatic Deployment:**
- Push to `main` → Render rebuilds containers
- Each service gets unique URL
- Environment variables set in Render dashboard

**See:** `/DEPLOYMENT.md` (parent directory)

---

### DigitalOcean App Platform

Use Docker Compose + DigitalOcean CLI.

**Steps:**
1. Create app.yaml from docker-compose.yml
2. Deploy with `doctl apps create --spec app.yaml`
3. Services auto-configured with load balancer

**See:** `/DEPLOYMENT.md` (parent directory)

---

### Self-Hosted (DigitalOcean Droplet)

Use docker-compose.yml directly on VPS.

**Setup:**
```bash
# On droplet
git clone ...
cd Real-Estate-Valuation

# Start
docker-compose up -d

# Monitor
docker-compose logs -f

# Stop
docker-compose down
```

---

## 🔐 Environment Variables

### Required (All Deployments)

```env
# Supabase (Database)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=sb_secret_xxxxx

# API Configuration
ENV=production
PORT=8000  # Auto-set by cloud platforms

# Streamlit Only
API_URL=https://api-url.com
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_ENABLEXSRFPROTECTION=false
```

### Optional

```env
DEBUG=false
LOG_LEVEL=info
WANDB_API_KEY=xxxxx  # For experiment tracking
```

---

## 🐛 Troubleshooting

### "Module not found" in container

**Cause:** Docker build context issue

**Fix:**
```bash
# Rebuild from scratch
docker-compose build --no-cache

# Or clean and rebuild
docker system prune -a
docker-compose build
```

### "Port already in use"

**Cause:** Port conflict with existing service

**Fix:**
```bash
# Use different port
docker-compose up -p 9000:8000

# Or kill existing process
docker ps  # Find container ID
docker kill <container-id>
```

### "Connection refused: localhost:8000"

**Cause:** API not accessible from Streamlit

**Fix (docker-compose):**
- Change `API_URL=http://localhost:8000` → `API_URL=http://api:8000`
- Services connect via internal Docker network

**Fix (deployed):**
- Use full deployed URL: `API_URL=https://api-domain.com`
- Not `localhost` or internal IP

### Logs show timeout errors

**Cause:** Models loading slow on first request

**Solution:**
- Render free tier: Models load on first request (~2-5 min)
- Wait for health check to pass
- Models cached after first load

---

## 📊 Build Process

### Multi-Stage Build (Dockerfile)

```
Stage 1: Builder
├── Python 3.11 slim
├── Install pip packages (in /usr/local/)
└── Creates clean layer

Stage 2: Runtime
├── Python 3.11 slim (fresh, no build tools)
├── Copy packages from builder
├── Copy app code
├── Create non-root user
└── Run
```

**Benefit:** Final image smaller (no build tools, compiler, etc.)

---

## ✅ Quick Reference

| Task | Command |
|------|---------|
| **Build images** | `docker-compose build` |
| **Start services** | `docker-compose up` |
| **View logs** | `docker-compose logs -f` |
| **Stop services** | `docker-compose down` |
| **Full cleanup** | `docker-compose down -v` |
| **Build API only** | `docker build -f .deployment/Dockerfile .` |
| **Run API** | `docker run -p 8000:8000 api:latest` |
| **Test container** | `docker run -it api:latest /bin/sh` |

---

## 📚 Related Documentation

- [DEPLOYMENT.md](../DEPLOYMENT.md) - Comprehensive deployment guide
- [README.md](../README.md) - Project overview
- [app/README.md](../app/README.md) - Backend architecture

---

## 🔍 File Cleanup History

**Removed (consolidation):**
- ❌ `Dockerfile.minimal` - Test version (unused)
- ❌ `Dockerfile.test` - Test version (unused)
- ❌ `railway.json` - Old platform config (using Render)
- ❌ `run_streamlit.py` - Duplicated by run.py
- ❌ `streamlit_start.sh` - Consolidated into start.sh

**Kept (production):**
- ✅ `Dockerfile` - Main API
- ✅ `Dockerfile.streamlit` - Streamlit UI
- ✅ `docker-compose.yml` - Local dev
- ✅ `.dockerignore` - Build optimization
- ✅ `start.sh` - Entrypoint script
- ✅ `run.py` - Streamlit launcher

---

**Last Updated:** 2026-07-23
**Docker Version:** 20.10+
**Compose Version:** 2.0+
