# -*- coding: utf-8 -*-
import subprocess
import time

import click

FEATURES = [
    "TI-19",
    #      "FI-16",
    #     "TI-33",
    #     "FI-2",
    #     "FI-151",
    #     "TI-8",
    #     "FI-241",
    #  "valve-position-12",  # dry-bed
    #     "FI-38",  # strippera
    #     "PI-28",  # stripper
    #     "TI-28",  # stripper
    #      "FI-20",
    #     "FI-30",
    "TI-3",
    "FI-19",
    #     "FI-211",
    "FI-11",
    #     "TI-30",
    #     "PI-30",
    "TI-1213",
    #     "TI-4",
    #    "FI-23",
    #    "FI-20",
    #   "FI-20/FI-23",
    #    "TI-22",
    #    "delta_t",
    "TI-35",
    #     "delta_t_2"
]

SLURM_SUBMISSION_TEMPLATE = """#!/bin/bash -l
#SBATCH --chdir ./
#SBATCH --mem       32GB
#SBATCH --ntasks    1
#SBATCH --cpus-per-task   4
#SBATCH --job-name  {name}
#SBATCH --time      72:00:00
#SBATCH --partition serial

source /home/kjablonk/anaconda3/bin/activate
conda activate timeseries

python run_scenarios.py {feature_a} {feature_b} {objective} {baseline}
"""


def write_submission_script(feature_a, feature_b, objective, baseline):
    submission_name = (
        f"new_scenario_{feature_a}_{feature_b}_{objective}_{baseline}".replace("/", "*")
    )
    script_content = SLURM_SUBMISSION_TEMPLATE.format(
        **{
            "name": submission_name,
            "feature_a": feature_a,
            "feature_b": feature_b,
            "objective": objective,
            "baseline": baseline,
        }
    )
    scriptname = f"{submission_name}.slurm"
    with open(scriptname, "w") as handle:
        handle.write(script_content)
    return scriptname


@click.command("cli")
@click.option("--submit", is_flag=True)
def run(submit):
    for baseline in ["full"]:
        for objective in ["amine", "co2"]:
            for i, feature_a in enumerate(FEATURES):
                for j, feature_b in enumerate(FEATURES):
                    if i < j:
                        filename = write_submission_script(
                            feature_a, feature_b, objective, baseline
                        )
                        if submit:
                            subprocess.call(f"sbatch {filename}", shell=True)
                            time.sleep(2)


if __name__ == "__main__":
    run()
