# syntax=docker/dockerfile:1.7
# Operator UI image (D-035): SvelteKit static build served by unprivileged nginx,
# which proxies /api and /ws to control-api. Build context = repo root.

FROM node:22-alpine AS build
ENV PNPM_HOME=/pnpm COREPACK_ENABLE_DOWNLOAD_PROMPT=0
RUN corepack enable
WORKDIR /src/frontend
COPY frontend/package.json frontend/pnpm-lock.yaml ./
RUN --mount=type=cache,id=fa-pnpm-store,target=/pnpm/store \
    pnpm install --frozen-lockfile
COPY frontend/ ./
RUN pnpm build

FROM nginxinc/nginx-unprivileged:1.29-alpine
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /src/frontend/build /usr/share/nginx/html
EXPOSE 8080
