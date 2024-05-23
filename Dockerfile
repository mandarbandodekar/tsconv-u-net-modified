FROM tensorflow/tensorflow:2.15.0-jupyter

WORKDIR /app

COPY . .  
RUN pip3 install -r requirements.txt
WORKDIR /app/src
EXPOSE 8000
EXPOSE 8888
ENTRYPOINT ["python3","train.py"]





