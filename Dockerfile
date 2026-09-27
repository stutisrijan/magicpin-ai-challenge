# Vera bot: single-process FastAPI service (state is in memory, so exactly one worker).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8080

WORKDIR /srv

RUN useradd --create-home --uid 10001 vera

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=vera:vera . .

USER vera
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/v1/healthz' % os.environ.get('PORT','8080'), timeout=4)" || exit 1

CMD sh -c "uvicorn bot:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1 --proxy-headers --timeout-keep-alive 75"
