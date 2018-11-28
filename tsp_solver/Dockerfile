FROM python:3

WORKDIR /usr/src/work

#COPY requirements.txt ./
#RUN pip install --no-cache-dir -r requirements.txt

RUN wget http://www.math.uwaterloo.ca/tsp/concorde/downloads/codes/src/co031219.tgz
RUN tar xf co031219.tgz
RUN cd concorde && CFLAGS='-O3 -march=native -mtune=native -fPIC' ./configure
RUN cd concorde/LINKERN && make -j && cp linkern ../../

COPY . .


CMD [ "./linkern", "-s", "42", "-S", "linkern.tour", "-R", "999999999", "-t", "10", "./cities1k.tsp"]