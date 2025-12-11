## Eternalore – Running (dev)
For a containerized setup (web + worker + Redis + Postgres):
```bash
1. pip install -r requirements.txt
2. docker compose up -d db redis worker
3. cd backend/core
4. python manage.py migrate --noinput 
5. python manage.py collectstatic --noinput 
6. celery -A core worker -l debug
7. export $(grep -v '^#' .env | xargs) \                                                           1 ↵
&& export DB_HOST=127.0.0.1 DB_PORT=5432 \
&& export CELERY_BROKER_URL=redis://127.0.0.1:6379/0 \
&& export CELERY_RESULT_BACKEND=redis://127.0.0.1:6379/1 \
&& export DEBUG=True \
&& python manage.py runserver 0.0.0.0:8001 --noreload


```
Then visit http://127.0.0.1:8000. Override environment with your own `.env` or pass env vars.
For testing you can grant yourself plus with:
```bash
docker compose exec web bash -lc "cd backend/core && python manage.py shell -c \"from user.models import User as U; u=U.objects.get(email='example@mail.com'); u.subscription_plan='plus'; u.save(); print(u.subscription_plan)\""
```
On native setup:
```
python manage.py shell -c "from user.models import User as U; u,_=U.objects.get_or_create(email='example@mail.com'); u.subscription_plan='plus'; u.save(update_fields=['subscription_plan']); print(u.subscription_plan)"
```
