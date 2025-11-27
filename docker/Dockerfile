# docker/Dockerfile

FROM python:3.12-slim

# Avoid Python buffering issues
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps for PyMuPDF, LanceDB, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libgl1-mesa-glx \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt

# Copy project code
COPY . /app

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
