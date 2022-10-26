FROM python:3.8-bullseye

WORKDIR /root

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt install -y python3-pip

COPY . .

RUN pip install -r requirements.txt

CMD ["python","train.py"]