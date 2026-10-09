FROM node:22-alpine AS build
WORKDIR /app
ARG VITE_API_BASE_URL=
ARG VITE_API_KEY=
ARG VITE_ADMIN_KEY=
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL \
    VITE_API_KEY=$VITE_API_KEY \
    VITE_ADMIN_KEY=$VITE_ADMIN_KEY
COPY package.json package-lock.json* ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
