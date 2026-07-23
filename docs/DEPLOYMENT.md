# 🚀 Deployment Guide

Guide để deploy Real Estate Valuation project lên các cloud platforms.

---

## 📋 Prerequisites

Trước khi deploy, chuẩn bị:

- **Git repository**: Code phải ở GitHub (hoặc platform khác)
- **Docker image**: Project có Dockerfile sẵn
- **Environment variables**: .env file với credentials
- **Models & Data**: Supabase database với dữ liệu training

---

## 🔧 Environment Variables

### Required Variables

```env
# Supabase (Database)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=sb_secret_xxxxx
SUPABASE_TABLE_CACHE=address_cache
SUPABASE_TABLE_FEATURES=Raw_Features
SUPABASE_TABLE_LISTINGS=listings

# API & Server
ENV=production
DEBUG=false
LOG_LEVEL=info
PORT=8000

# Streamlit (Optional)
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_ENABLEXSRFPROTECTION=false
API_URL=https://your-api-domain.com
```

---

## 1️⃣ Render (Current)

### Quick Setup

1. **Connect GitHub to Render**
   - Go to https://dashboard.render.com
   - Click "New" → "Web Service"
   - Select GitHub repo

2. **Configure API Service**
   ```
   Name: real-estate-valuation-api
   Runtime: Docker
   Dockerfile: ./Dockerfile
   Port: 8000
   ```

3. **Set Environment Variables**
   - Environment → Add all variables from `.env`

4. **Deploy Streamlit (Optional)**
   ```
   Name: real-estate-valuation-streamlit
   Runtime: Docker
   Dockerfile: .deployment/Dockerfile.streamlit
   Port: 8501
   Environment: Add API_URL + Supabase vars
   ```

### Limitations (Free Tier)
- ⚠️ Slow startup (models load on first request)
- ⚠️ Auto-spins down after 15 min inactivity
- ⚠️ Limited disk space (~10GB)
- ⚠️ Limited memory (512MB)

---

## 2️⃣ DigitalOcean App Platform

### Step 1: Create App

```bash
# Install doctl CLI
brew install digitalocean/digitalocean/doctl

# Authenticate
doctl auth init

# Create app from GitHub
doctl apps create --spec app.yaml
```

### Step 2: Create `app.yaml`

```yaml
name: real-estate-valuation
services:
  - name: api
    github:
      repo: your-username/Real-Estate-Valuation
      branch: main
    build_command: |
      pip install -r requirements.txt
    run_command: |
      python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT
    http_port: 8000
    envs:
      - key: ENV
        value: production
      - key: SUPABASE_URL
        scope: RUN_AND_BUILD_TIME
        value: ${DB_SUPABASE_URL}
      - key: SUPABASE_SERVICE_KEY
        scope: RUN_AND_BUILD_TIME
        value: ${DB_SUPABASE_SERVICE_KEY}
    # ... other env vars

  - name: streamlit
    github:
      repo: your-username/Real-Estate-Valuation
      branch: main
    build_command: |
      pip install streamlit
    run_command: |
      streamlit run app/ui/streamlit_app.py --server.port $PORT
    http_port: 8501
    envs:
      - key: API_URL
        value: https://${API_DOMAIN}/api

databases:
  - name: redis
    engine: REDIS
    version: "7"

static_sites:
  - name: docs
    source_dir: docs
```

### Step 3: Deploy

```bash
doctl apps update [app-id] --spec app.yaml
```

### DigitalOcean Advantages
- ✅ Better performance than Render
- ✅ Persistent storage (100GB+)
- ✅ More memory (2GB+)
- ✅ No auto-shutdown
- ✅ Easy scaling
- 💰 ~$5-12/month for basic app

---

## 3️⃣ DigitalOcean Droplet (Self-Hosted)

### Most Control & Cheapest

### Step 1: Create Droplet

```bash
# SSH into droplet
ssh root@your-droplet-ip

# Update system
apt update && apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose
apt install docker-compose -y
```

### Step 2: Create `docker-compose.yml`

```yaml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - ENV=production
      - SUPABASE_URL=${SUPABASE_URL}
      - SUPABASE_SERVICE_KEY=${SUPABASE_SERVICE_KEY}
      - PORT=8000
    restart: always
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  streamlit:
    build:
      context: .
      dockerfile: .deployment/Dockerfile.streamlit
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://api:8000
      - SUPABASE_URL=${SUPABASE_URL}
      - SUPABASE_SERVICE_KEY=${SUPABASE_SERVICE_KEY}
    depends_on:
      - api
    restart: always

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - api
      - streamlit
    restart: always
```

### Step 3: Deploy

```bash
# Copy code to droplet
scp -r . root@your-droplet-ip:/app

# SSH vào
ssh root@your-droplet-ip
cd /app

# Create .env file
nano .env
# Paste environment variables

# Start services
docker-compose up -d

# View logs
docker-compose logs -f
```

### Step 4: Setup SSL (Let's Encrypt)

```bash
# Install certbot
apt install certbot python3-certbot-nginx -y

# Get certificate
certbot certonly --standalone -d your-domain.com

# Configure nginx to use SSL
# (Update nginx.conf with certificate paths)

# Restart
docker-compose restart nginx
```

### Droplet Advantages
- ✅ Cheapest (~$5/month)
- ✅ Full control
- ✅ No vendor lock-in
- ✅ Easy to backup
- ❌ Requires manual management
- ❌ Need to handle security updates

---

## 4️⃣ AWS EC2

### Deploy with ECS/Fargate

### Step 1: Create ECR Repository

```bash
aws ecr create-repository --repository-name real-estate-valuation

# Get login token
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin [your-account-id].dkr.ecr.us-east-1.amazonaws.com

# Build & push
docker build -t real-estate-valuation .
docker tag real-estate-valuation:latest [account-id].dkr.ecr.us-east-1.amazonaws.com/real-estate-valuation:latest
docker push [account-id].dkr.ecr.us-east-1.amazonaws.com/real-estate-valuation:latest
```

### Step 2: Create ECS Cluster & Task

```bash
# Create cluster
aws ecs create-cluster --cluster-name real-estate

# Register task definition (from ECS console or CLI)
aws ecs register-task-definition --cli-input-json file://task-definition.json

# Create service
aws ecs create-service \
  --cluster real-estate \
  --service-name api \
  --task-definition real-estate-valuation \
  --desired-count 1
```

### AWS Advantages
- ✅ Enterprise-grade
- ✅ Auto-scaling
- ✅ Load balancing
- ✅ RDS for databases
- ❌ Complex setup
- ❌ Expensive (~$50+/month)

---

## 🔄 CI/CD Pipeline

### GitHub Actions (Free)

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Build Docker image
        run: |
          docker build -t real-estate-valuation .
          docker tag real-estate-valuation [registry]/real-estate-valuation:latest

      - name: Push to registry
        run: |
          docker push [registry]/real-estate-valuation:latest

      - name: Deploy to Render
        run: |
          curl -X POST https://api.render.com/deploy/srv-[service-id] \
            -H "Authorization: Bearer ${{ secrets.RENDER_DEPLOY_KEY }}"
```

---

## 📊 Comparison Table

| Platform | Cost | Startup | Control | Setup Difficulty |
|----------|------|---------|---------|------------------|
| **Render** | Free/Paid | Slow | Low | Easy |
| **DigitalOcean App** | $5-12/mo | Fast | Medium | Medium |
| **Droplet** | $5-20/mo | Fast | High | Hard |
| **AWS EC2** | $10-50+/mo | Fast | High | Hard |
| **Heroku** | $7-50+/mo | Medium | Low | Easy |

---

## 🛠️ Troubleshooting

### "Module not found"
- Ensure `requirements.txt` has all dependencies
- Check Docker COPY commands

### "Port already in use"
- Kill existing process: `lsof -i :8000`
- Or use different port

### "Supabase connection failed"
- Verify credentials in `.env`
- Check firewall rules
- Test: `curl https://your-project.supabase.co`

### "Slow API startup"
- First request loads ML models (normal)
- Use warm-up requests in health checks
- Consider pre-warming in startup script

### "Out of memory"
- Reduce model batch size
- Use swap on Droplet: `fallocate -l 2G /swapfile`
- Scale up to larger instance

---

## 📝 Best Practices

1. **Use environment variables** - Never hardcode secrets
2. **Health checks** - Add `/health` endpoint for monitoring
3. **Logging** - Use structured logging for debugging
4. **Backups** - Regularly backup Supabase data
5. **Monitoring** - Setup alerts for failures
6. **Auto-restart** - Use `restart: always` in Docker
7. **SSL/TLS** - Always use HTTPS in production
8. **Rate limiting** - Protect API from abuse

---

## 🔒 Security Checklist

- [ ] All secrets in environment variables
- [ ] HTTPS enabled
- [ ] CORS configured properly
- [ ] Rate limiting enabled
- [ ] Database backups working
- [ ] Monitoring & alerts setup
- [ ] Firewall rules configured
- [ ] Regular security updates

---

## 📞 Support

- **Render**: https://render.com/docs
- **DigitalOcean**: https://docs.digitalocean.com
- **AWS**: https://docs.aws.amazon.com
- **Docker**: https://docs.docker.com

---

**Last Updated**: 2026-07-23
