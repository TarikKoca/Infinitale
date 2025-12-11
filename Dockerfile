FROM python:3.11-slim AS py

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential libpq-dev curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build Tailwind CSS (node via npx is invoked from package.json)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get update && apt-get install -y nodejs && \
    npm ci --omit=dev || npm i && \
    npm run build:css && \
    apt-get purge -y nodejs npm && apt-get autoremove -y && rm -rf /var/lib/apt/lists/* /root/.npm

EXPOSE 8000


