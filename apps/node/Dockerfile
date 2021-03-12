FROM hivecell/pysyft:latest

RUN mkdir /app
WORKDIR /app

COPY requirements.txt entrypoint.sh /app/
COPY /src /app/src
COPY /demo /app/demo

RUN pip3 install -r requirements.txt 

RUN chmod +x /app/demo/data/MNIST/download.sh \
 && sh /app/demo/data/MNIST/download.sh

WORKDIR /app/

ENTRYPOINT ["sh", "entrypoint.sh"]
