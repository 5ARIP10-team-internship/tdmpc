# Temporal Difference Learning for Model Predictive Control

Original PyTorch implementation of **TD-MPC** from

[Temporal Difference Learning for Model Predictive Control](https://arxiv.org/abs/2203.04955) by

[Nicklas Hansen](https://nicklashansen.github.io), [Xiaolong Wang](https://xiaolonw.github.io)\*, [Hao Su](https://cseweb.ucsd.edu/~haosu)\*


<p align="center">
  <br><img src='media/ae591483.png' width="600"/><br>
   <a href="https://arxiv.org/abs/2203.04955">[Paper]</a>&emsp;<a href="https://nicklashansen.github.io/td-mpc">[Website]</a>
</p>

## Method

**TD-MPC** is a framework for model predictive control (MPC) using a Task-Oriented Latent Dynamics (TOLD) model and a terminal value function *learned jointly* by temporal difference (TD) learning. TD-MPC plans actions entirely in latent space using the TOLD model, which learns compact task-centric representations from either state or image inputs. TD-MPC solves challenging Humanoid and Dog locomotion tasks in 1M environment steps.

## Getting Started

__Requirements__

- Python >= 3.9

---

__Installation__

Navigate to `https://pytorch.org/get-started/locally/` and install the PyTorch (version 2.6.0) with Cuda. Next, clone and enter this repository:
```
git@github.com:5ARIP10-team-internship/tdmpc.git
cd tdmpc
```
Create a virtual environment and install dependencies:
```
python -m venv tdmpc-env --system-site-packages
source tdmpc-env/bin/activate
pip install -r requirements.txt
```

---

__Training and testing__

After installing dependencies, you can train an agent by calling
```
python src/train.py task=Pendulum-v1
```

Model weights can be saved with the argument `save_model=True`. Refer to the `cfgs` directory for a full list of options and default hyperparameters, and see `tasks.txt` for a list of supported tasks.

The training script supports both local logging as well as cloud-based logging with [Weights & Biases](https://wandb.ai). To use W&B, provide a key by setting the environment variable `WANDB_API_KEY=<YOUR_KEY>` and add your W&B project and entity details to `cfgs/default.yaml`.
