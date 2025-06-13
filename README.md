# Maritime Field Document Extraction App

This project provides a very small scale deployment for extracting structured data from maritime field documents. It consists of a FastAPI backend and a Next.js frontend. Data is stored locally in the `uploads/` directory and containers are orchestrated via Docker Compose.

## Development

1. Copy `.env.example` to `.env` and add your `OPENAI_API_KEY`.
2. Build and start services:

```bash
docker compose up -d --build
```

The frontend will be available at http://localhost:3000 and proxies API requests to the backend.

## Updating

Pull the latest changes and rebuild:

```bash
git pull
docker compose down
docker compose up -d --build
```
