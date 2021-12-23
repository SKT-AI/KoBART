# NSMC classification example

## Run on a GPU machine

- build a docker image

    ```bash
    docker build -t kobart -f docker/Dockerfile .
    ```

- run a docker container

    ```bash
    cd ~/KoBART # root directory of this repository
    docker run --rm -it \
        -v $PWD:/home/ubuntu/KoBART \
        -e PYTHONPATH="/home/ubuntu/KoBART" \
        -w "/home/ubuntu/KoBART/examples" \
        --name "kobart" \
        kobart /bin/sh
    ```

- finetune KoBART model with NSMC

  - :warning: run on the docker container

  - finetune

    ```bash
    python nsmc.py --gpus 1 --max_epochs 3 --default_root_dir .cache --gradient_clip_val 1.0
    ```
