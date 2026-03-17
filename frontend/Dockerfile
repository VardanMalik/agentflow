# ---------- builder ----------
FROM node:20-alpine AS builder

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

# ---------- runtime ----------
FROM nginx:alpine AS runtime

LABEL org.opencontainers.image.title="AgentFlow Frontend" \
      org.opencontainers.image.description="Distributed AI Agent Orchestration Platform - Dashboard" \
      org.opencontainers.image.source="https://github.com/agentflow/agentflow" \
      org.opencontainers.image.licenses="MIT"

COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["wget", "-qO-", "http://localhost/"]

CMD ["nginx", "-g", "daemon off;"]
