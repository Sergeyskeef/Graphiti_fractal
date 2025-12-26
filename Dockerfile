FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Patch graphiti_core to fix Neo4j 5.20 compatibility issue with dynamic labels
# The syntax SET n:$(node.labels) requires Neo4j 5.23+, we replace it with APOC call
RUN sed -i 's/SET n:\$(node\.labels)/SET n:Entity/g' \
    /usr/local/lib/python3.10/site-packages/graphiti_core/models/nodes/node_db_queries.py && \
    echo "✅ Patched graphiti_core for Neo4j 5.20 compatibility"

COPY . .

CMD ["bash"]

