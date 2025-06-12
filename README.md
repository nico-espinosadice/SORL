<div align="center">

<div id="user-content-toc" style="margin-bottom: 50px">
  <ul align="center" style="list-style: none;">
    <summary>
      <h1>Scaling Offline RL via <br> Efficient and Expressive Shortcut Models</h1>
      <br>
      <h2><a href="https://arxiv.org/abs/2505.22866">Paper</a> | <a href="https://nico-espinosadice.github.io/projects/sorl/">Project Page</a>< | <a href="https://x.com/nico_espinosa_d/status/1933209680609788170">Thread</a> /h2>
    </summary>
  </ul>
</div>

<img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+">
<img src="https://img.shields.io/badge/linter-Ruff-blueviolet.svg" alt="Linter: Ruff">

</div>

## Installation

SORL requires Python 3.9+ and is based on JAX. The main dependencies are
`jax >= 0.4.26`, `ogbench == 1.1.0`, and `gymnasium == 0.29.1`.
To install the full dependencies, simply run:
```bash
pip install -r requirements.txt
```

## Usage

The main implementation of SORL is in [agents/sorl.py](agents/sorl.py). 
```bash
# SORL on OGBench scene-play (offline RL)
python main.py --env_name=scene-play-singletask-v0 --agent.q_loss_coefficient=100
```

## Reproducing Results

Following [FQL](https://github.com/seohongpark/fql)'s official implementation, we provide the complete list of the **exact command-line flags**
used to produce the main results of SORL in the paper. Note that we follow the official implementation of [FQL](https://github.com/seohongpark/fql) for usage of `q_agg` and `discount`. In other words, we set `q_agg` and `discount` based on how the FQL baseline uses them. We use seeds 1-8 for the results in our paper. 

> **Note:** In OGBench, each environment provides five tasks, one of which is the default task.
> This task corresponds to the environment ID without any task suffixes.
> For example, the default task of `antmaze-large-navigate` is `task1`,
> and `antmaze-large-navigate-singletask-v0` is the same environment as `antmaze-large-navigate-singletask-task1-v0`.

<details>
<summary><b>Full list of commands</b></summary>

#### SORL on state-based OGBench (default tasks)

```bash
# SORL on OGBench antmaze-large-navigate-singletask-v0 (=antmaze-large-navigate-singletask-task1-v0)
python main.py --env_name=antmaze-large-navigate-singletask-v0 --agent.q_agg=min --agent.q_loss_coefficient=500
# SORL on OGBench antmaze-giant-navigate-singletask-v0 (=antmaze-giant-navigate-singletask-task1-v0)
python main.py --env_name=antmaze-giant-navigate-singletask-v0 --agent.discount=0.995 --agent.q_agg=min --agent.q_loss_coef=500
# SORL on OGBench humanoidmaze-medium-navigate-singletask-v0 (=humanoidmaze-medium-navigate-singletask-task1-v0)
python main.py --env_name=humanoidmaze-medium-navigate-singletask-v0 --agent.discount=0.995 --agent.q_loss_coef=100
# SORL on OGBench humanoidmaze-large-navigate-singletask-v0 (=humanoidmaze-large-navigate-singletask-task1-v0)
python main.py --env_name=humanoidmaze-large-navigate-singletask-v0 --agent.discount=0.995 --agent.q_loss_coef=500
# SORL on OGBench antsoccer-arena-navigate-singletask-v0 (=antsoccer-arena-navigate-singletask-task4-v0)
python main.py --env_name=antsoccer-arena-navigate-singletask-v0 --agent.q_loss_coef=500
# SORL on OGBench cube-single-play-singletask-v0 (=cube-single-play-singletask-task2-v0)
python main.py --env_name=cube-single-play-singletask-v0 --agent.q_loss_coef=10
# SORL on OGBench cube-double-play-singletask-v0 (=cube-double-play-singletask-task2-v0)
python main.py --env_name=cube-double-play-singletask-v0 --agent.q_loss_coef=50
# SORL on OGBench scene-play-singletask-v0 (=scene-play-singletask-task2-v0)
python main.py --env_name=scene-play-singletask-v0 --agent.q_loss_coef=100
```

#### SORL on state-based OGBench (all tasks)

```bash
# SORL on OGBench antmaze-large-navigate-singletask-{task1, task2, task3, task4, task5}-v0 (default: task1)
python main.py --env_name=antmaze-large-navigate-singletask-task1-v0 --agent.q_agg=min --agent.q_loss_coef=500
python main.py --env_name=antmaze-large-navigate-singletask-task2-v0 --agent.q_agg=min --agent.q_loss_coef=500
python main.py --env_name=antmaze-large-navigate-singletask-task3-v0 --agent.q_agg=min --agent.q_loss_coef=500
python main.py --env_name=antmaze-large-navigate-singletask-task4-v0 --agent.q_agg=min --agent.q_loss_coef=500
python main.py --env_name=antmaze-large-navigate-singletask-task5-v0 --agent.q_agg=min --agent.q_loss_coef=500
# SORL on OGBench antmaze-giant-navigate-singletask-{task1, task2, task3, task4, task5}-v0 (default: task1)
python main.py --env_name=antmaze-giant-navigate-singletask-task1-v0 --agent.discount=0.995 --agent.q_agg=min --agent.q_loss_coef=500
python main.py --env_name=antmaze-giant-navigate-singletask-task2-v0 --agent.discount=0.995 --agent.q_agg=min --agent.q_loss_coef=500
python main.py --env_name=antmaze-giant-navigate-singletask-task3-v0 --agent.discount=0.995 --agent.q_agg=min --agent.q_loss_coef=500
python main.py --env_name=antmaze-giant-navigate-singletask-task4-v0 --agent.discount=0.995 --agent.q_agg=min --agent.q_loss_coef=500
python main.py --env_name=antmaze-giant-navigate-singletask-task5-v0 --agent.discount=0.995 --agent.q_agg=min --agent.q_loss_coef=500
# SORL on OGBench humanoidmaze-medium-navigate-singletask-{task1, task2, task3, task4, task5}-v0 (default: task1)
python main.py --env_name=humanoidmaze-medium-navigate-singletask-task1-v0 --agent.discount=0.995 --agent.q_loss_coef=100
python main.py --env_name=humanoidmaze-medium-navigate-singletask-task2-v0 --agent.discount=0.995 --agent.q_loss_coef=100
python main.py --env_name=humanoidmaze-medium-navigate-singletask-task3-v0 --agent.discount=0.995 --agent.q_loss_coef=100
python main.py --env_name=humanoidmaze-medium-navigate-singletask-task4-v0 --agent.discount=0.995 --agent.q_loss_coef=100
python main.py --env_name=humanoidmaze-medium-navigate-singletask-task5-v0 --agent.discount=0.995 --agent.q_loss_coef=100
# SORL on OGBench humanoidmaze-large-navigate-singletask-{task1, task2, task3, task4, task5}-v0 (default: task1)
python main.py --env_name=humanoidmaze-large-navigate-singletask-task1-v0 --agent.discount=0.995 --agent.q_loss_coef=500
python main.py --env_name=humanoidmaze-large-navigate-singletask-task2-v0 --agent.discount=0.995 --agent.q_loss_coef=500
python main.py --env_name=humanoidmaze-large-navigate-singletask-task3-v0 --agent.discount=0.995 --agent.q_loss_coef=500
python main.py --env_name=humanoidmaze-large-navigate-singletask-task4-v0 --agent.discount=0.995 --agent.q_loss_coef=500
python main.py --env_name=humanoidmaze-large-navigate-singletask-task5-v0 --agent.discount=0.995 --agent.q_loss_coef=500
# SORL on OGBench antsoccer-arena-navigate-singletask-{task1, task2, task3, task4, task5}-v0 (default: task4)
python main.py --env_name=antsoccer-arena-navigate-singletask-task1-v0 --agent.discount=0.995 --agent.q_loss_coef=500
python main.py --env_name=antsoccer-arena-navigate-singletask-task2-v0 --agent.discount=0.995 --agent.q_loss_coef=500
python main.py --env_name=antsoccer-arena-navigate-singletask-task3-v0 --agent.discount=0.995 --agent.q_loss_coef=500
python main.py --env_name=antsoccer-arena-navigate-singletask-task4-v0 --agent.discount=0.995 --agent.q_loss_coef=500
python main.py --env_name=antsoccer-arena-navigate-singletask-task5-v0 --agent.discount=0.995 --agent.q_loss_coef=500
# SORL on OGBench cube-single-play-singletask-{task1, task2, task3, task4, task5}-v0 (default: task2)
python main.py --env_name=cube-single-play-singletask-task1-v0 --agent.q_loss_coef=10
python main.py --env_name=cube-single-play-singletask-task2-v0 --agent.q_loss_coef=10
python main.py --env_name=cube-single-play-singletask-task3-v0 --agent.q_loss_coef=10
python main.py --env_name=cube-single-play-singletask-task4-v0 --agent.q_loss_coef=10
python main.py --env_name=cube-single-play-singletask-task5-v0 --agent.q_loss_coef=10
# SORL on OGBench cube-double-play-singletask-{task1, task2, task3, task4, task5}-v0 (default: task2)
python main.py --env_name=cube-double-play-singletask-task1-v0 --agent.q_loss_coef=50
python main.py --env_name=cube-double-play-singletask-task2-v0 --agent.q_loss_coef=50
python main.py --env_name=cube-double-play-singletask-task3-v0 --agent.q_loss_coef=50
python main.py --env_name=cube-double-play-singletask-task4-v0 --agent.q_loss_coef=50
python main.py --env_name=cube-double-play-singletask-task5-v0 --agent.q_loss_coef=50
# SORL on OGBench scene-play-singletask-{task1, task2, task3, task4, task5}-v0 (default: task2)
python main.py --env_name=scene-play-singletask-task1-v0 --agent.q_loss_coef=100
python main.py --env_name=scene-play-singletask-task2-v0 --agent.q_loss_coef=100
python main.py --env_name=scene-play-singletask-task3-v0 --agent.q_loss_coef=100
python main.py --env_name=scene-play-singletask-task4-v0 --agent.q_loss_coef=100
python main.py --env_name=scene-play-singletask-task5-v0 --agent.q_loss_coef=100
```

</details>


## Acknowledgments

This codebase is built on top of [FQL](https://github.com/seohongpark/fql)'s official implementation, which adapts [OGBench](https://github.com/seohongpark/ogbench)'s reference implementations.

<!-- ## Citation
```bibtex
@misc{,
      title={}, 
      author={},
      year={},
      eprint={},
      archivePrefix={},
      primaryClass={}
}
``` -->
