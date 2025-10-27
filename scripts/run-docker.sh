#!/bin/bash
set -euo pipefail

# Secure and unified Docker runner script
# Usage: ./scripts/run-container.sh <environment>
# Environments: development, staging, production

if [ $# -ne 1 ]; then
  echo "Usage: $0 <environment>"
  echo "Environments: development, staging, production"
  exit 1
fi

ENV=$1

# Validate environment argument
if [[ ! "$ENV" =~ ^(development|staging|production)$ ]]; then
  echo "❌ Invalid environment: $ENV"
  echo "Must be one of: development, staging, production"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$PROJECT_ROOT/.env.$ENV"

# Load environment variables if the env file exists
if [ -f "$ENV_FILE" ]; then
  echo "🔧 Loading environment variables from $ENV_FILE"
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
else
  echo "⚠️  Warning: $ENV_FILE not found. Continuing with existing environment variables."
fi

cd "$PROJECT_ROOT"

# Build Docker image name dynamically
IMAGE_NAME="myapp:${ENV}"

echo "🐳 Building Docker image for $ENV environment..."
docker build \
  --build-arg APP_ENV="$ENV" \
  -t "$IMAGE_NAME" .

echo "🚀 Starting Docker containers for $ENV environment..."

# Run docker compose with proper env file if present
if [ -f "$ENV_FILE" ]; then
  docker compose --env-file "$ENV_FILE" up -d --build db app
else
  docker compose up -d --build db app
fi

# For development, automatically mount source code for hot reload
if [ "$ENV" = "development" ]; then
  echo "🔁 Development mode: enabling hot reload and mounting source code..."
  docker run -it --rm \
    -p 8000:8000 \
    -v "$PROJECT_ROOT":/app \
    --env-file "$ENV_FILE" \
    "$IMAGE_NAME"
else
  echo "✅ $ENV environment containers are up and running."
fi
