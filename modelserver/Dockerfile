FROM nvidia/cuda:11.0-cudnn8-runtime-ubuntu18.04-rc

RUN apt-get update && apt-get install python3 python3-pip -y

RUN pip3 install --upgrade pip

RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt /app/

RUN pip3 install -r /app/requirements.txt

COPY . /app

CMD ["/usr/bin/python3", "-u", "/app/main.py"]
