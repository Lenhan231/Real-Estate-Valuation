# Production Deployment Guide

## Overview

This guide covers deploying the Real Estate Valuation API to production using Railway and Supabase.

## Architecture

```
┌─────────────────────────────────────────┐
│       Frontend (Optional)               │
│  - Vercel (Next.js/React)               │
│  - Streamlit (app/ui/streamlit_app.py)  │
└──────────────┬──────────────────────────┘
               │ HTTPS API calls
┌──────────────▼──────────────────────────┐
│    FastAPI Backend (Railway)            │
│  - /api/predict                         │
│  - /api/feedback                        │
│  - /api/admin                           │
│  - /health (monitoring)                 │
└──────────────┬──────────────────────────┘
               │ PostgreSQL queries
┌──────────────▼──────────────────────────┐
│    Supabase (Database + Auth)           │
│  - PostgreSQL database                  │
│  - Feedback storage                     │
│  - User authentication (optional)       │
└─────────────────────────────────────────┘
```

## Prerequisites

1. **Railway Account**: Sign up at [railway.app](https://railway.app)
2. **Supabase Project**: Already configured (check SUPABASE_URL in .env)
3. **GitHub Account**: For connecting your repository
4. **Git**: For version control

## Step 1: Prepare Repository

### 1.1 Add Production Files

All deployment files have been created:
- `Dockerfile` - Container image definition
- `railway.json` - Railway configuration
- `.dockerignore` - Files to exclude from Docker build
- `.env.example` - Environment variables template

### 1.2 Verify Dependencies

```bash
# Check that all required packages are in requirements.txt
cat requirements.txt | grep -E "fastapi|uvicorn|gunicorn"

# Output should show:
# fastapi
# uvicorn
# (other dependencies)
```

## Step 2: Deploy to Railway

### 2.1 Connect GitHub Repository

1. Go to [railway.app](https://railway.app) and sign up/login
2. Click **"Create New Project"** → **"Deploy from GitHub"**
3. Select your repository: `Real-Estate-Valuation`
4. Authorize Railway to access your GitHub

### 2.2 Configure Environment Variables

After connecting, Railway detects the Dockerfile automatically.

1. Go to your project → **Variables** tab
2. Add the following variables:

```env
# Copy from your local .env file
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-key-here
ENV=production
DEBUG=false
LOG_LEVEL=info
```

**Never commit .env to Git!** Railway stores secrets securely.

### 2.3 Deploy

Railway automatically deploys when you push to `main` branch:

```bash
# After making changes locally:
git add .
git commit -m "chore: add production deployment files"
git push origin main

# Railway will:
# 1. Detect Dockerfile
# 2. Build container image
# 3. Run health checks
# 4. Deploy to production
```

### 2.4 Verify Deployment

1. Go to Railway dashboard → Your project → **Deployments**
2. Wait for status to show **"Success"** (usually 2-5 minutes)
3. Check the generated URL: `https://your-project-xxxx.railway.app`

Test the API:
```bash
curl https://your-project-xxxx.railway.app/health

# Expected response:
# {"status":"healthy","service":"Real Estate Valuation API","version":"2.4.0"}
```

## Step 3: Connect Supabase

### 3.1 Verify Supabase Credentials

Make sure your Railway environment has:
- `SUPABASE_URL` (your project URL)
- `SUPABASE_SERVICE_KEY` (admin key from Settings → API)

### 3.2 Create Feedback Table (if needed)

If the `feedback` table doesn't exist in Supabase:

```sql
CREATE TABLE feedback (
  id SERIAL PRIMARY KEY,
  predicted_price_vnd BIGINT NOT NULL,
  actual_price_vnd BIGINT,
  rating INT,
  bucket VARCHAR(50),
  confidence FLOAT,
  feature_version INT DEFAULT 1,
  features_json JSONB,
  timestamp TIMESTAMP DEFAULT NOW(),
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_feedback_timestamp ON feedback(timestamp);
CREATE INDEX idx_feedback_bucket ON feedback(bucket);
```

## Step 4: Monitor Production

### 4.1 Health Checks

Railway automatically monitors:
```
GET /health - Should return 200 OK every 30 seconds
```

### 4.2 View Logs

In Railway dashboard:
1. Select your project
2. Click **"Logs"** tab
3. Filter by service name or search for errors

### 4.3 Set Up Alerts (Optional)

1. Go to **Settings** → **Notifications**
2. Enable email alerts for:
   - Deployment failures
   - Service crashes
   - Health check failures

## Step 5: Connect Frontend (Optional)

### 5.1 Update API URLs

If you have a frontend (Next.js, React, etc.):

```javascript
// .env.production
REACT_APP_API_URL=https://your-project-xxxx.railway.app

// Usage in components:
fetch(`${process.env.REACT_APP_API_URL}/api/predict`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(predictionData)
})
```

### 5.2 Deploy to Vercel (Optional)

```bash
# If using Next.js, deploy to Vercel:
npm install -g vercel
vercel

# Follow prompts to connect GitHub and deploy
```

## Step 6: Update Streamlit (Optional)

If you want to run Streamlit app against production:

```bash
# Update your Streamlit app to use production API:
# app/ui/streamlit_app.py

API_URL = "https://your-project-xxxx.railway.app"

# Then deploy Streamlit:
streamlit run app/ui/streamlit_app.py --server.sslCertFile=/path/to/cert
```

Or deploy Streamlit to Railway as well:

```bash
# Create streamlit_config.toml
[server]
headless = true
port = 8501

# Push to Railway
git push origin main
```

## Troubleshooting

### Issue: Deployment Failed

**Check logs:**
```bash
# In Railway dashboard: Logs tab
# Look for error messages like:
# - "Module not found"
# - "Port already in use"
# - "Environment variable missing"
```

**Common fixes:**
1. Verify all environment variables are set
2. Check that all dependencies in `requirements.txt` are compatible
3. Ensure Dockerfile has correct Python version (3.11)

### Issue: Health Check Failing

```bash
# Test locally:
python -c "import requests; requests.get('http://localhost:8000/health')"

# If fails, check:
# 1. Is FastAPI running? (uvicorn app.main:app)
# 2. Is /health endpoint defined? (check app/main.py)
# 3. Are Supabase credentials valid?
```

### Issue: 502 Bad Gateway

Usually means the app crashed:
1. Check logs for Python errors
2. Verify memory allocation (Railway provides 512MB default)
3. Check model loading in `app/core/models.py`

### Issue: Database Connection Errors

```
# Error: "could not translate host name"
# Solution: Verify SUPABASE_URL is correct

# Error: "permission denied"
# Solution: Check SUPABASE_SERVICE_KEY has admin privileges
```

## Performance Monitoring

### 4.1 Track Predictions

Add metrics to track in production:

```python
# In app/routers/predict.py
import time

@router.post("/predict")
async def predict(request: PredictRequest):
    start_time = time.time()
    
    # ... prediction logic ...
    
    duration = time.time() - start_time
    print(f"[PERF] Prediction took {duration:.2f}s")
```

### 4.2 Model Performance Tracking

Monitor predictions vs actual prices:

```bash
# Query Supabase feedback table:
SELECT 
  DATE(timestamp) as date,
  COUNT(*) as prediction_count,
  AVG(ABS(predicted_price_vnd - actual_price_vnd) / actual_price_vnd) as mape
FROM feedback
GROUP BY DATE(timestamp)
ORDER BY date DESC;
```

## Maintenance

### Regular Tasks

**Weekly:**
- Check logs for errors
- Monitor model performance metrics

**Monthly:**
- Review feedback data quality
- Check if retraining is needed
  ```bash
  POST /api/admin/retrain
  ```
- Update dependencies if security patches available

**Quarterly:**
- Retrain models with new feedback
- Test disaster recovery procedures
- Review and update documentation

### Auto-Retraining (Optional)

Set up a cron job to retrain models:

```bash
# Using GitHub Actions (create .github/workflows/retrain.yml)
name: Monthly Model Retraining
on:
  schedule:
    - cron: '0 0 1 * *'  # First day of month at midnight UTC

jobs:
  retrain:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Trigger Retraining
        run: |
          curl -X POST ${{ secrets.RAILWAY_API_URL }}/api/admin/retrain \
            -H "Authorization: Bearer ${{ secrets.API_KEY }}"
```

## Cost Estimation

**Railway:**
- Free tier: Up to $5/month credit
- Pay-as-you-go after that: ~$0.50/hour for small app

**Supabase:**
- Free tier: Good for development
- Pro tier: ~$25/month for production

**Total estimated cost:** $25-50/month

## Next Steps

1. ✅ Push deployment files to GitHub
2. ⏳ Connect Railway to repository
3. ⏳ Set environment variables
4. ⏳ Deploy and test
5. ⏳ Monitor production logs
6. ⏳ Set up monitoring/alerts
7. ⏳ Configure auto-retraining

## Support

For deployment issues:
- Railway Docs: https://docs.railway.app
- FastAPI Docs: https://fastapi.tiangolo.com
- Supabase Docs: https://supabase.com/docs

---

**Last updated:** 2026-07-23
**Maintained by:** Real Estate Valuation Team
