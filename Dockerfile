# The image runs the conversation pipeline in text mode, which is everything a
# container can run: there is no microphone and no speaker behind it.
#
# That is why only requirements/base.txt is installed. The audio extras pull in
# torch, transformers and spaCy — several gigabytes — to drive hardware the
# container does not have, and PortAudio, ALSA headers and a C compiler to
# build against. Leaving them out takes the image from multiple gigabytes to a
# few hundred megabytes and removes the build toolchain entirely.

# --- build ------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# A virtualenv rather than the system site-packages, so the runtime stage can
# take the dependencies as one directory and nothing else.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements/base.txt /tmp/requirements/base.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements/base.txt

# --- runtime ----------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH" \
    VOICE_IO=text \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

COPY --from=builder /opt/venv /opt/venv

RUN useradd --create-home --uid 1000 assistant
WORKDIR /app

COPY --chown=assistant:assistant . .
# /app itself is created root-owned by WORKDIR, so anything that writes a new
# file beside the code — pytest's cache, a dotenv written at runtime — fails
# with EACCES even though every copied file is owned correctly.
RUN mkdir -p /app/logs && chown -R assistant:assistant /app

USER assistant
EXPOSE 8501

# Streamlit's own readiness endpoint. Checked with urllib rather than curl so
# the image needs no extra package.
HEALTHCHECK --interval=15s --timeout=5s --start-period=25s --retries=5 \
  CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4).status == 200 else 1)"

CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]

# --- test -------------------------------------------------------------------
# Built on demand, never shipped:
#   docker build --target test -t ai-voice-assistant:test . && \
#   docker run --rm ai-voice-assistant:test
FROM runtime AS test

USER root
COPY requirements/dev.txt /tmp/requirements/dev.txt
COPY requirements/base.txt /tmp/requirements/base.txt
RUN pip install --no-cache-dir -r /tmp/requirements/dev.txt
USER assistant

CMD ["python", "-m", "pytest", "tests/"]
