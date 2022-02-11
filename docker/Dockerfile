FROM nvcr.io/nvidia/pytorch:21.05-py3

WORKDIR $HOME/KoBART/examples

RUN pip install pytorch-lightning==1.2.1 transformers==4.3.3 boto3

ENTRYPOINT [ "/bin/sh", "-c", "/bin/bash" ]
