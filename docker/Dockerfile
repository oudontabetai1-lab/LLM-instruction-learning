# データ生成・キュレーション・評価用イメージ(学習は GPU ホスト上で [train] extras を使用)
FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config
COPY data/seed_tasks.jsonl ./data/seed_tasks.jsonl

RUN pip install --no-cache-dir -e ".[aws]"

ENTRYPOINT ["itl"]
CMD ["--help"]
