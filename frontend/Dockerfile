FROM node:20-alpine AS builder
WORKDIR /app
COPY package.json tsconfig.json next.config.js ./
RUN npm install --omit=dev
COPY app ./app
RUN npm run build

FROM node:20-alpine
WORKDIR /app
COPY --from=builder /app .
EXPOSE 3000
CMD ["npm", "start"]
