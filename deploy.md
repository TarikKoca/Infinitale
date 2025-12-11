## Production Deployment – Eternalore

This guide covers a clean, reproducible production deployment using Docker, Docker Compose, Gunicorn (ASGI), Celery workers/beat, Redis, Postgres, and Nginx with optional Let's Encrypt.

### 1) Prerequisites
- Linux host (2 vCPU+, 4GB RAM recommended)
- Docker 24+ and Docker Compose v2
- A domain pointing to the host's public IP
- Open ports 80 and 443 on the host firewall

### 2) Clone and prepare
```bash
git clone https://github.com/your-org/Eternalore.git /opt/eternalore
cd /opt/eternalore

# Copy production environment template
cp backend/core/env.production.example backend/core/.env

# Edit with your secrets and domain
$EDITOR backend/core/.env
```

Required keys (non-exhaustive): `SECRET_KEY`, `ALLOWED_HOSTS`, `DB_PASSWORD`, SMTP credentials (optional), Sentry DSN (optional).

### 3) One-time file system prep
```bash
mkdir -p certbot/conf certbot/www
mkdir -p backend/core/var/media/image backend/core/var/media/audio
mkdir -p backend/core/var/logs
```

### 4) Build and run (first deploy)
```bash
docker compose -f docker-compose.prod.yml build --no-cache
docker compose -f docker-compose.prod.yml up -d db redis

# Wait for db/redis to be healthy, then bring up app stack
docker compose -f docker-compose.prod.yml up -d web worker beat nginx

# Create a Django superuser (interactive)
docker compose -f docker-compose.prod.yml exec web bash -lc "cd backend/core && python manage.py createsuperuser"
```

The `web` container automatically runs migrations and collectstatic before starting Gunicorn.

### 5) TLS/HTTPS (Let's Encrypt)
Option A: Obtain certs on the host (recommended for simplicity):
```bash
sudo apt-get update && sudo apt-get install -y certbot
sudo certbot certonly --standalone -d yourdomain.com -d www.yourdomain.com

# Copy certs into mapped folder for nginx container
sudo rsync -a /etc/letsencrypt/ certbot/conf/
```

Update `nginx/sites/default.conf` to replace `yourdomain.com` with your domain. Then reload:
```bash
docker compose -f docker-compose.prod.yml restart nginx
```

Renewal (cron):
```bash
sudo crontab -e
# 0 3 * * * certbot renew --quiet && rsync -a /etc/letsencrypt/ /opt/eternalore/certbot/conf/ && docker compose -f /opt/eternalore/docker-compose.prod.yml restart nginx
```

### 6) Environment flags for security
Ensure in `backend/core/.env`:
- DEBUG=False
- Proper `ALLOWED_HOSTS`
- `SESSION_COOKIE_SECURE=True`, `CSRF_COOKIE_SECURE=True`
- `SECURE_SSL_REDIRECT=True`
- `SECURE_HSTS_SECONDS=31536000` (after confirming HTTPS works)
- `CSRF_TRUSTED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com`

### 7) Operational commands
```bash
# Logs
docker compose -f docker-compose.prod.yml logs -f web
docker compose -f docker-compose.prod.yml logs -f worker
docker compose -f docker-compose.prod.yml logs -f beat
docker compose -f docker-compose.prod.yml logs -f nginx

# Status
docker compose -f docker-compose.prod.yml ps

# Exec into web container
docker compose -f docker-compose.prod.yml exec web bash

# Django manage.py commands
docker compose -f docker-compose.prod.yml exec web bash -lc "cd backend/core && python manage.py migrate"
docker compose -f docker-compose.prod.yml exec web bash -lc "cd backend/core && python manage.py collectstatic --noinput"

# Scale workers
docker compose -f docker-compose.prod.yml up -d --scale worker=2 worker
```

### 8) Health checks
The app exposes:
- `/health/` basic
- `/health/readiness/` readiness
- `/health/liveness/` liveness

Compose uses these for `web` health. Nginx also checks HTTP health.

### 9) Backups (example)
```bash
BACKUP_DIR=/opt/eternalore/backups
mkdir -p "$BACKUP_DIR"
cat > /opt/eternalore/backup.sh <<'EOS'
#!/usr/bin/env bash
set -euo pipefail
cd /opt/eternalore
DATE=$(date +%Y%m%d_%H%M%S)
docker compose -f docker-compose.prod.yml exec -T db pg_dump -U ${DB_USER:-Eternalore} ${DB_NAME:-Eternalore} > "$BACKUP_DIR/db_${DATE}.sql"
find "$BACKUP_DIR" -name 'db_*.sql' -mtime +30 -delete
EOS
chmod +x /opt/eternalore/backup.sh

sudo crontab -e
# 15 2 * * * /opt/eternalore/backup.sh
```

### 10) Updating
```bash
cd /opt/eternalore
git pull origin main
docker compose -f docker-compose.prod.yml build
docker compose -f docker-compose.prod.yml up -d
docker compose -f docker-compose.prod.yml exec web bash -lc "cd backend/core && python manage.py migrate"
```

### 11) Notes
- Static files are served by Nginx from the `static_volume` mounted at `/static`.
- Media is mounted read-only to Nginx at `/media` and read-write to `web`/`worker`.
- For S3 media, set `USE_S3_MEDIA=True` and provide AWS variables.
- Gunicorn is configured to use ASGI workers via UvicornWorker.

You're done. Visit https://yourdomain.com

### Staging environment

Use a separate VM or the same host with a separate compose file.

1) DNS: point `staging.yourdomain.com` to the staging host.

2) Env:
```bash
cp backend/core/env.staging.example backend/core/.env
$EDITOR backend/core/.env
```

3) Bring up staging:
```bash
docker compose -f docker-compose.staging.yml build --no-cache
docker compose -f docker-compose.staging.yml up -d db-staging redis-staging
docker compose -f docker-compose.staging.yml up -d web-staging worker-staging beat-staging nginx-staging
docker compose -f docker-compose.staging.yml exec web-staging bash -lc "cd backend/core && python manage.py createsuperuser"
```

4) TLS: obtain Let’s Encrypt certs for `staging.yourdomain.com`, copy into `certbot/conf`, and restart nginx:
```bash
docker compose -f docker-compose.staging.yml restart nginx-staging
```

5) Safety:
- Use sandbox credentials for payments and external services
- Keep separate DB/Redis volumes from production
- Consider disallowing indexing for staging


