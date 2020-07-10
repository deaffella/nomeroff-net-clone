FROM tensorflow/tensorflow:1.15.2-py3
ENV EBIAN_FRONTEND noninteractive
ENV TZ=Europe/Kiev
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN mkdir -p /var/www
WORKDIR /var/www

RUN apt-get update

# For opencv
RUN apt-get install -y libglib2.0

# For Mask_RCNN
RUN apt-get install -y libsm6
RUN apt-get install -y libfontconfig1 libxrender1
RUN apt-get install -y libxtst6
RUN apt-get install -y git
RUN apt-get install -y nano
RUN apt-get install -y wget

RUN pip3 install scikit_image
RUN pip3 install Cython
RUN pip3 install pycocotools
RUN pip3 install matplotlib
RUN pip3 install tensorflow==1.15.*
RUN pip3 install opencv_python
RUN pip3 install imutils
RUN pip3 install "numpy>=1.16.*"
RUN pip3 install keras==2.2.*
RUN pip3 install jupyter
RUN pip3 install imgaug
RUN pip3 install asyncio
RUN pip3 install GitPython
RUN pip3 install pycocotools
RUN pip3 install tqdm
RUN pip3 install git+https://github.com/matterport/Mask_RCNN

RUN cd /var/www/ && wget "https://github.com/ria-com/nomeroff-net/raw/master/examples/images/example1.jpeg" -O 1.JPG && git clone https://github.com/ria-com/nomeroff-net.git
#&& cd nomeroff-net && python3 setup.py install
WORKDIR /var/www/nomeroff-net
