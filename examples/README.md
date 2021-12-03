# NSMC classification example

## Run on a GPU machine

### Prepare

- pull a nvidia docker image

    ```bash
    docker pull nvcr.io/nvidia/pytorch:21.05-py3
    ```

- run a docker container

    ```bash
    cd ~/KoBART # root directory of this repository
    docker run --rm -d -it \
        -v $PWD:/home/ubuntu/KoBART \
        -e PYTHONPATH="/home/ubuntu/KoBART" \
        -w "/home/ubuntu/KoBART/examples" \
        --name "kobart-nsmc" \
        --entrypoint="/bin/bash" \
        nvcr.io/nvidia/pytorch:21.05-py3
    ```

### Run

- finetune KoBART model with NSMC
  - :warning: run on the docker container

    ```bash
    docker exec -it kobart-nsmc bash
    ```

  - install python packages

    ```bash
    pip install pytorch-lightning==1.2.1 transformers==4.3.3 wget
    ```

  - finetune

    ```bash
    python nsmc.py --gpus 1 --max_epochs 3 --default_root_dir .cache --gradient_clip_val 1.0
    ```
