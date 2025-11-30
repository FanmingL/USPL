# Uncertainty-Sensitive Privileged Learning

Uncertainty-Sensitive Privileged Learning (USPL) is a privileged learning framework designed for partially observable decision-making problems. Traditional privileged learning frameworks often fail to enable policies to learn active information-gathering behaviors regarding privileged information. Consequently, while policies may perform well during training (when privileged information is available), their performance degrades during deployment due to the absence of this information.

USPL utilizes an observation encoder to predict unobservable privileged information from historical observations while simultaneously estimating the confidence of the current prediction. Based on this confidence, the USPL policy decides whether to utilize the predicted privileged information for decision-making or to further explore to gather information. Across 9 tasks, including quadruped robot navigation and drone hovering, USPL achieved a success rate exceeding 95%, significantly outperforming traditional privileged learning and pure reinforcement learning baselines.

## Environment Dependencies

We provide a Docker environment capable of running the training. You can pull the Docker image using the following command:

```bash
docker pull core.116.172.93.164.nip.ip:30670/public/luofanming:20250423004526
````

Users can also install the Python dependencies based on `requirements.txt`.

## Training

First, start the Docker container from the root directory of the current code:

```bash
docker run --rm -it -v $PWD:/home/ubuntu/workspace --gpus all core.116.172.93.164.nip.ip:30670/public/luofanming:20250423004526 /bin/bash
```

Then, install `legged_gym`:

```bash
cd /home/ubuntu/workspace/legged_gym
pip install -e .
```

Return to the code root directory and start training:

```bash
cd /home/ubuntu/workspace
python gen_tmuxp.py
tmuxp load run_all.json
```

Training logs can be found in the `logfile` directory. Training parameters can be modified in `gen_tmuxp.py`; for instance, if you need to train on a different environment, simply modify the value corresponding to `task_name`.

## Citation

```
@inproceedings{luo2025uspl,
  title={Uncertainty-Sensitive Privileged Learning},
  author={Luo, Fan-Ming and Yuan, Lei and Yu, Yang},
  booktitle={Advances in Neural Information Processing Systems 39},
  address = {San Diego, CA},
  year={2025}
}
```