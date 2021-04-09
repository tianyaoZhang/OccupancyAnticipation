#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import wandb

from habitat_baselines.common.baseline_registry import baseline_registry
from occant_baselines.config.default import get_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval"],
        required=True,
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()
    run_exp(**vars(args))

def mkdir_ifnotexists(directory):
    tempDir=""
    for dr in directory.split('/'):
        tempDir=os.path.join(tempDir,dr)
        if not os.path.exists(tempDir):
            os.mkdir(tempDir)


def run_exp(exp_config: str, run_type: str, opts=None) -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.

    Returns:
        None.
    """

    # # 2. Save model inputs and hyperparameters
    # wandb_config = wandb.config
    # wandb_config.dropout = 0.01
    print('--> shell command : {0}'.format(' '.join(sys.argv)))
    config = get_config(exp_config, opts)
    # 与baseline兼容
    id = wandb.util.generate_id()
    try:
        config.defrost()
        try:
            config.EXPERIMENT_NAME
        except:
            config.EXPERIMENT_NAME=f'EXPS/baseline/{config.RL.ANS.OCCUPANCY_ANTICIPATOR.type}'
            # mkdir_ifnotexists(config.EXPERIMENT_NAME)
        mkdir_ifnotexists(f"{config.EXPERIMENT_NAME}")
        if config.USE_TIMESTAMP:
            print("--> using timestamp")
            timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
            # if os.path.exists(os.path.join(config.EXPERIMENT_NAME)):
            if config.CONTINUE:
                print("--> continue")
                timestamps = os.listdir(config.EXPERIMENT_NAME)
                if (len(timestamps)) > 0:
                    timestamp = sorted(timestamps)[-1]
                    id = os.listdir(f'{config.EXPERIMENT_NAME}/'
                                    f'{timestamp}/wandb/latest-run')[1].split('-')[-1].split('.')[0]
            # mkdir_ifnotexists(f"{config.EXPERIMENT_NAME}/{timestamp}")
            config.EXPERIMENT_NAME=f"{config.EXPERIMENT_NAME}/{timestamp}"

        mkdir_ifnotexists(config.EXPERIMENT_NAME)
        config.TENSORBOARD_DIR =f"{config.EXPERIMENT_NAME}/{config.TENSORBOARD_DIR}"
        config.VIDEO_DIR = f"{config.EXPERIMENT_NAME}/{config.VIDEO_DIR}"
        config.CHECKPOINT_FOLDER = f"{config.EXPERIMENT_NAME}/{config.CHECKPOINT_FOLDER.split('/')[-1]}"
        config.WANDB_ID = id
        config.freeze()
    except:
        print('--> except try')
        config.freeze()
        pass

    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)

    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"

    # id='176fdo1y'
    project_name='OccAnt-Explore'

    wandb.init(project=project_name, monitor_gym=True, resume="allow",
               config=config, name=timestamp , id=id, reinit=True, #sync_tensorboard=True,
               dir=f'{config.EXPERIMENT_NAME}', job_type=run_type)

    print(f"--> [id]  = {id}")

    trainer = trainer_init(config)

    os.system("""cp -r {0} "{1}" """.format(exp_config,
                                            f"{config.EXPERIMENT_NAME}/"
                                            f"{os.path.basename(exp_config)}"))
    with open(f"{config.EXPERIMENT_NAME}/full_config.yaml",'w') as df:
        df.write(f"{config}")
    with open(f"{config.EXPERIMENT_NAME}/run.sh",'w') as f:
        f.write("python "+' '.join(sys.argv))
        f.write(f"# wandb info:\n#project={project_name}\t"
                f"id={id}\tjob_type={run_type}")

    if run_type == "train":
        trainer.train()
    elif run_type == "eval":
        trainer.eval()


if __name__ == "__main__":
    main()
