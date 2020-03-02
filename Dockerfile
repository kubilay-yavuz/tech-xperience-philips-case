FROM python:3

ADD test.py /

RUN pip install keras
RUN pip install pandas
RUN pip install numpy
RUN pip install opencv-python

CMD [ "python", "./test.py" ]