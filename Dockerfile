FROM docker.io/nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

RUN apt-get update \
&& apt-get install -y python3-venv python3-pip sshfs \
&& useradd -ms /bin/bash cc

# switch user
USER cc

ENV PATH /home/cc/.local/bin:${PATH}

RUN mkdir -p /home/cc/.local/bin

# install connectors
RUN python3 -m venv /home/cc/.local/red \
&& . /home/cc/.local/red/bin/activate \
&& pip install wheel \
&& pip install red-connector-ssh==1.2 \
&& ln -s /home/cc/.local/red/bin/red-connector-* /home/cc/.local/bin

# install app
RUN pip3 install --user numpy torch torchvision matplotlib scikit-image pillow

ADD --chown=cc:cc model.py /home/cc/.local/bin/model.py
ADD --chown=cc:cc utils.py /home/cc/.local/bin/utils.py
ADD --chown=cc:cc train.py /home/cc/.local/bin/train.py
ADD --chown=cc:cc validate.py /home/cc/.local/bin/validate.py
ADD --chown=cc:cc main.py /home/cc/.local/bin/main.py