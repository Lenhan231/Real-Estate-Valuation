# 🚀 Production Deployment - Quick Start

## What's Ready

✅ **Docker containerization** - Dockerfile with multi-stage build  
✅ **Railway configuration** - railway.json ready for deployment  
✅ **Health checks** - Automatic monitoring endpoint  
✅ **Logging** - Production-grade structured logging  
✅ **Environment setup** - .env.example with all variables  
✅ **Local testing** - docker-compose for development  

---

## 🎯 Deploy in 5 Minutes

### 1️⃣ Create Railway Account
Go to [railway.app](https://railway.app) and sign up with GitHub

### 2️⃣ Connect Repository
1. Click **"Create New Project"**
2. Select **"Deploy from GitHub"**
3. Choose **Real-Estate-Valuation**
4. Authorize Railway

### 3️⃣ Set Environment Variables

Add these in Railway dashboard (Settings → Variables):

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-key-from-supabase
ENV=production
DEBUG=false
LOG_LEVEL=info
```

⚠️ **Never commit .env to Git!**

### 4️⃣ Deploy
Push to main branch:
```bash
git push origin main
```

Railway automatically:
- Detects Dockerfile
- Builds container (2-5 min)
- Runs health checks
- Deploys to production

### 5️⃣ Test Deployment
```bash
# Your deployed URL will look like:
# https://real-estate-valuation-xxxx.railway.app

# Test health check:
curl https://your-url.railway.app/health

# Expected response:
{
  "status": "healthy",
  "service": "Real Estate Valuation API",
  "version": "2.4.0",
  "timestamp": "2026-07-23T...",
  "environment": "production",
  "database": "connected",
  "feedback_count": 42
}
```

---

## 📊 Monitor Your Deployment

### In Railway Dashboard
- **Logs** - Real-time application logs
- **Deployments** - Deployment history
- **Metrics** - CPU, memory, network usage
- **Alerts** - Email notifications for errors

### Health Check
- Runs every 30 seconds
- Automatically restarts if failed
- Includes database connectivity check

### View Logs
```bash
# In Railway: Logs tab
# Or via CLI:
railway logs --follow
```

---

## 🔧 Troubleshooting

### Deployment Failed?
1. Check **Logs** tab in Railway
2. Look for error messages
3. Verify environment variables are set
4. Check that Dockerfile exists

### Health Check Failing?
1. Verify `SUPABASE_URL` is correct
2. Verify `SUPABASE_SERVICE_KEY` has admin access
3. Check Supabase project is active

### API Not Responding?
1. Check that health check is passing
2. Verify port is 8000
3. Look at error logs

---

## 📈 Next Steps

After successful deployment:

1. **Update API URL** in any frontend/client apps
   ```javascript
   const API_URL = "https://your-url.railway.app"
   ```

2. **Test endpoints**
   ```bash
   # Prediction endpoint
   curl -X POST https://your-url.railway.app/api/predict \
     -H "Content-Type: application/json" \
     -d '{"street": "Test", "locality": "Phường 1", ...}'

   # Feedback endpoint
   curl -X POST https://your-url.railway.app/api/feedback \
     -H "Content-Type: application/json" \
     -d '{"predicted_price_vnd": 5000000000, ...}'
   ```

3. **Monitor in production**
   - Check logs daily for errors
   - Track prediction counts
   - Monitor model accuracy with feedback data

4. **Set up auto-retraining** (optional)
   - Monthly retraining with: `POST /api/admin/retrain`
   - Monitor drift with: `GET /api/admin/drift-status`

---

## 💰 Costs

- **Railway**: $5/month free credit (covers most small apps)
- **Supabase**: Free tier for development ($25/month for production)
- **Total**: ~$25-50/month

---

## 📚 Full Documentation

See [DEPLOYMENT.md](DEPLOYMENT.md) for:
- Detailed setup instructions
- Database schema
- Frontend integration
- Monitoring setup
- Maintenance procedures
- Troubleshooting guide

---

## ✨ Features Deployed

✅ **Real-time predictions** with ensemble models  
✅ **Feedback collection** for model improvement  
✅ **Automatic model retraining** with drift detection  
✅ **Explainable AI** with feature importance  
✅ **Health monitoring** with auto-restart  
✅ **Production logging** for debugging  
✅ **API documentation** at `/docs`  

---

**Need help?**
- Railway Docs: https://docs.railway.app
- FastAPI Docs: https://fastapi.tiangolo.com
- Check logs: Railway Dashboard → Logs tab

---

**Status**: ✅ Ready to deploy
**Version**: 2.4.0
**Last Updated**: 2026-07-23
