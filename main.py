# main.py
import os
import platform
import sys

import json
import random
import time
import copy

import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset, ReplayBuffer
from utils.evaluation import evaluate, flatten
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import (
    CsvLogger,
    get_exp_name,
    get_flag_dict,
    get_wandb_video,
    setup_wandb,
)

import flax
from flax.core import FrozenDict

FLAGS = flags.FLAGS

flags.DEFINE_string("wandb_project", "sorl", "wandb project")
flags.DEFINE_string("run_group", "Debug", "Run group.")
flags.DEFINE_integer("seed", 1, "Random seed.")
flags.DEFINE_string(
    "env_name", "scene-play-singletask-task1-v0", "Environment (dataset) name."
)
flags.DEFINE_string("save_dir", "exp/", "Save directory.")
flags.DEFINE_string("restore_path", None, "Restore path.")
flags.DEFINE_integer("restore_epoch", None, "Restore epoch.")

flags.DEFINE_integer("offline_steps", 1_000_000, "Number of offline steps.")
flags.DEFINE_integer("online_steps", 0, "Number of online steps.")
flags.DEFINE_integer("buffer_size", 2_000_000, "Replay buffer size.")
flags.DEFINE_integer("log_interval", 5_000, "Logging interval.")
flags.DEFINE_integer("validation_log_interval", 25_000, "Validation logging interval.")
flags.DEFINE_integer("eval_interval", 100_000, "Evaluation interval.")
flags.DEFINE_integer("save_interval", 100_000, "Saving interval.")

flags.DEFINE_integer("eval_episodes", 50, "Number of evaluation episodes.")
flags.DEFINE_integer("video_episodes", 0, "Number of video episodes for each task.")
flags.DEFINE_integer("video_frame_skip", 3, "Frame skip for videos.")

flags.DEFINE_float("p_aug", None, "Probability of applying image augmentation.")
flags.DEFINE_integer("frame_stack", None, "Number of frames to stack.")
flags.DEFINE_integer(
    "balanced_sampling", 0, "Whether to use balanced sampling for online fine-tuning."
)
flags.DEFINE_bool("track", True, "Whether to track experiments with Weights & Biases.")
flags.DEFINE_string("unique_id", "00000", "Unique ID")
flags.DEFINE_string("sweep_id", "0000", "Sweep ID")
flags.DEFINE_bool(
    "debug", False, "debug mode avoids first round evaluation to run faster"
)

config_flags.DEFINE_config_file("agent", "agents/sorl.py", lock_config=False)


# run via: python main.py --agent=agents/sorl.py
def main(_):
    print("running agent name", FLAGS.agent.agent_name)

    # Set up logger.
    exp_name = get_exp_name(FLAGS.seed, FLAGS.agent.agent_name)
    mode = "disabled" if not FLAGS.track else "online"
    setup_wandb(
        project=FLAGS.wandb_project, group=FLAGS.run_group, name=exp_name, mode=mode
    )

    FLAGS.save_dir = os.path.join(
        FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name
    )
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, "flags.json"), "w") as f:
        json.dump(flag_dict, f)

    # Make environment and datasets.
    config = FLAGS.agent
    if config["agent_name"] == "sorlbc":
        config["offline_steps"] = FLAGS.offline_steps

    env, eval_env, train_dataset, val_dataset = make_env_and_datasets(
        FLAGS.env_name, frame_stack=FLAGS.frame_stack
    )
    if FLAGS.video_episodes > 0:
        assert "singletask" in FLAGS.env_name, (
            "Rendering is currently only supported for OGBench environments."
        )
    if FLAGS.online_steps > 0:
        assert "visual" not in FLAGS.env_name, (
            "Online fine-tuning is currently not supported for visual environments."
        )

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Set up datasets.
    train_dataset = Dataset.create(**train_dataset)
    if FLAGS.balanced_sampling:
        # Create a separate replay buffer so that we can sample from both the training dataset and the replay buffer.
        example_transition = {k: v[0] for k, v in train_dataset.items()}
        replay_buffer = ReplayBuffer.create(example_transition, size=FLAGS.buffer_size)
    else:
        # Use the training dataset as the replay buffer.
        train_dataset = ReplayBuffer.create_from_initial_dataset(
            dict(train_dataset), size=max(FLAGS.buffer_size, train_dataset.size + 1)
        )
        replay_buffer = train_dataset
    # Set p_aug and frame_stack.
    for dataset in [train_dataset, val_dataset, replay_buffer]:
        if dataset is not None:
            dataset.p_aug = FLAGS.p_aug
            dataset.frame_stack = FLAGS.frame_stack
            if config["agent_name"] == "rebrac":
                dataset.return_next_actions = True

    # Create agent.
    example_batch = train_dataset.sample(1)

    agent_class = agents[config["agent_name"]]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch["observations"],
        example_batch["actions"],
        config,
    )

    # Restore agent.
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # Train agent.
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, "train.csv"))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, "eval.csv"))
    first_time = time.time()
    last_time = time.time()

    step = 0
    done = True
    expl_metrics = dict()
    online_rng = jax.random.PRNGKey(FLAGS.seed)
    for i in tqdm.tqdm(
        range(1, FLAGS.offline_steps + FLAGS.online_steps + 1),
        smoothing=0.1,
        dynamic_ncols=True,
    ):
        if i <= FLAGS.offline_steps:
            start_time = time.time()
            # Offline RL.
            batch = train_dataset.sample(config["batch_size"])

            if config["agent_name"] == "rebrac":
                agent, update_info = agent.update(
                    batch, full_update=(i % config["actor_freq"] == 0)
                )
            else:
                agent, update_info = agent.update(batch)
            stop_time = time.time()
        else:
            # Online fine-tuning.
            online_rng, key = jax.random.split(online_rng)

            if done:
                step = 0
                ob, _ = env.reset()

            action = agent.sample_actions(observations=ob, temperature=1, seed=key)
            action = np.array(action)

            next_ob, reward, terminated, truncated, info = env.step(action.copy())
            done = terminated or truncated

            if "antmaze" in FLAGS.env_name and (
                "diverse" in FLAGS.env_name
                or "play" in FLAGS.env_name
                or "umaze" in FLAGS.env_name
            ):
                # Adjust reward for D4RL antmaze.
                reward = reward - 1.0

            replay_buffer.add_transition(
                dict(
                    observations=ob,
                    actions=action,
                    rewards=reward,
                    terminals=float(done),
                    masks=1.0 - terminated,
                    next_observations=next_ob,
                )
            )
            ob = next_ob

            if done:
                expl_metrics = {
                    f"exploration/{k}": np.mean(v) for k, v in flatten(info).items()
                }

            step += 1

            # Update agent.
            if FLAGS.balanced_sampling:
                # Half-and-half sampling from the training dataset and the replay buffer.
                dataset_batch = train_dataset.sample(config["batch_size"] // 2)
                replay_batch = replay_buffer.sample(config["batch_size"] // 2)
                batch = {
                    k: np.concatenate([dataset_batch[k], replay_batch[k]], axis=0)
                    for k in dataset_batch
                }
            else:
                batch = train_dataset.sample(config["batch_size"])

            if config["agent_name"] == "rebrac":
                agent, update_info = agent.update(
                    batch, full_update=(i % config["actor_freq"] == 0)
                )
            else:
                agent, update_info = agent.update(batch)

        # Log metrics.
        if i % FLAGS.log_interval == 0:
            train_metrics = {f"{k}": v for k, v in update_info.items()}
            if i % FLAGS.validation_log_interval == 0:
                if val_dataset is not None:
                    val_batch = val_dataset.sample(config["batch_size"])
                    _, val_info = agent.total_loss(val_batch, grad_params=None)
                    train_metrics.update(
                        {f"validation/{k}": v for k, v in val_info.items()}
                    )
            train_metrics["time/first_last_over_log_time"] = (
                time.time() - last_time
            ) / FLAGS.log_interval
            train_metrics["time/total_time"] = time.time() - first_time
            train_metrics["time/epoch_time"] = stop_time - start_time
            train_metrics.update(expl_metrics)
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        if FLAGS.eval_interval != 0 and (
            (not FLAGS.debug and i == 1) or i % FLAGS.eval_interval == 0
        ):
            eval_metrics = {}
            renders = []

            if "sorl" in config["agent_name"]:
                max_log = int(np.log2(agent.config["denoise_timesteps"]))
                for log_k in range(max_log + 1):
                    k = 2**log_k

                    for n_val in agent.config["best_of_n_sweep"]:
                        eval_start_time = time.time()
                        print(f"Eval: k={k}, n={n_val}")

                        eval_info, trajs, cur_renders = evaluate(
                            agent=agent,
                            env=eval_env,
                            num_eval_episodes=FLAGS.eval_episodes,
                            num_video_episodes=FLAGS.video_episodes,
                            video_frame_skip=FLAGS.video_frame_skip,
                            sample_actions_kwargs=dict(
                                inference_timesteps=k,
                                n=n_val,
                            ),
                        )

                        renders.extend(cur_renders)
                        for key, val in eval_info.items():
                            eval_total_time = time.time() - eval_start_time
                            eval_metrics[f"time/k={k}/n={n_val}/total_eval_time"] = (
                                eval_total_time
                            )

                            total_states = sum(
                                len(traj["observation"]) for traj in trajs
                            )
                            if total_states > 0:
                                eval_metrics[
                                    f"time/k={k}/n={n_val}/eval_time_per_state"
                                ] = eval_total_time / total_states
                            if k == 1:
                                eval_metrics[f"evaluation/n={n_val}/{key}"] = val
                            else:
                                eval_metrics[
                                    f"eval_inference_{str(k)}/n={n_val}/{key}"
                                ] = val
            else:
                eval_start_time = time.time()
                renders = []
                eval_metrics = {}

                eval_info, trajs, cur_renders = evaluate(
                    agent=agent,
                    env=eval_env,
                    config=config,
                    num_eval_episodes=FLAGS.eval_episodes,
                    num_video_episodes=FLAGS.video_episodes,
                    video_frame_skip=FLAGS.video_frame_skip,
                )

                renders.extend(cur_renders)
                for k, v in eval_info.items():
                    eval_metrics[f"evaluation/{k}"] = v

                # Log total evaluation time
                eval_total_time = time.time() - eval_start_time
                eval_metrics["time/total_eval_time"] = eval_total_time

                # Calculate total number of states in trajectories
                total_states = sum(len(traj["observation"]) for traj in trajs)
                if total_states > 0:
                    eval_metrics["time/eval_time_per_state"] = (
                        eval_total_time / total_states
                    )

            if FLAGS.video_episodes > 0:
                video = get_wandb_video(renders=renders)
                eval_metrics["video"] = video

            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        # Save agent.
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

    train_logger.close()
    eval_logger.close()


if __name__ == "__main__":
    app.run(main)
