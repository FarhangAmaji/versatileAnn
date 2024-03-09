FROM pytorch/pytorch

ENV PYTHONUNBUFFERED 1

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

WORKDIR /app/tests
