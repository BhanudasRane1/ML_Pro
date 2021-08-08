FROM centos:latest

RUN yum install python36 -y

RUN pip3 install Flask

RUN pip3 install scikit-learn

RUN pip3 install pandas

RUN pip3 install numpy

WORKDIR /app

COPY . /app

EXPOSE 4000

CMD python3 app.py
