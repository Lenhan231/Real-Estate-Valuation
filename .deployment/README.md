# Deployment Configuration

This directory contains all deployment-related files for the Real Estate Valuation API.

## Files

- **Dockerfile** - Multi-stage Docker image definition for production deployment
- **docker-compose.yml** - Local development setup with optional Streamlit service
- **railway.json** - Configuration for Railway.app deployment
- **.dockerignore** - Files excluded from Docker build context

## Quick Start

### Local Deployment (Docker Compose)

```bash
# Start API only
docker-compose up -d

# Start API with Streamlit dashboard
docker-compose --profile with-streamlit up -d
```

### Railway Deployment

```bash
# Push to main branch to trigger automatic deployment
git push origin main
```

## Documentation

See parent directory for deployment guides:
- `../docs/DEPLOYMENT_QUICK_START.md` - 5-minute quick start
- `../docs/DEPLOYMENT.md` - Comprehensive deployment guide

## Environment Variables

Required environment variables (set in deployment platform):
- `SUPABASE_URL` - Your Supabase project URL
- `SUPABASE_SERVICE_KEY` - Supabase service role key
- `ENV` - Environment (production/development)
- `DEBUG` - Debug mode (true/false)
- `LOG_LEVEL` - Logging level (info/debug/warning)
- `PORT` - Server port (auto-set by cloud platforms)
