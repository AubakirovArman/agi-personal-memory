FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml .
RUN pip install --no-cache-dir -e . && pip install --no-cache-dir pytest

COPY src/ src/
COPY tests/ tests/

ENV AGIM_HOME=/data
VOLUME /data

EXPOSE 8720

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "from agim.core.system import AGIMSystem; AGIMSystem(workdir='/data')" || exit 1

CMD ["python", "-m", "agim.cli.main", "api", "--host", "0.0.0.0", "--port", "8720"]
