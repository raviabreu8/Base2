import csv
import math
import os

from stable_baselines3.common.callbacks import BaseCallback


class TrainingRewardCallback(BaseCallback):
    """
    Guarda apenas a recompensa total de cada episódio de treino.

    O Stable-Baselines3 adiciona info['episode']['r'] quando um
    episódio termina através do wrapper Monitor.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose=verbose)
        self.episode_rewards = []

    def _on_step(self):
        infos = self.locals.get("infos", [])

        for info in infos:
            episode_info = info.get("episode")

            if episode_info is not None:
                self.episode_rewards.append(
                    float(episode_info["r"])
                )

        return True

    def get_reward_summary(self, window_size=10):
        """
        Calcula a média dos primeiros e últimos episódios.
        """

        if not self.episode_rewards:
            return float("nan"), float("nan"), float("nan")

        window = min(
            int(window_size),
            len(self.episode_rewards)
        )

        start_values = self.episode_rewards[:window]
        end_values = self.episode_rewards[-window:]

        mean_reward_start = sum(start_values) / len(start_values)
        mean_reward_end = sum(end_values) / len(end_values)
        reward_improvement = mean_reward_end - mean_reward_start

        return (
            float(mean_reward_start),
            float(mean_reward_end),
            float(reward_improvement)
        )


class SimpleResultsCSV:
    """
    Guarda uma única linha por experiência num único results.csv.
    """

    FIELDNAMES = [
        "experiment",
        "model",
        "activation",
        "environment",
        "ablation",
        "training_timesteps",
        "evaluation_episodes",
        "mean_covered_area_m2",
        "total_collisions",
        "total_falls",
        "mean_reward_start",
        "mean_reward_end",
        "reward_improvement"
    ]

    def __init__(self, csv_path):
        self.csv_path = os.path.abspath(csv_path)

        csv_directory = os.path.dirname(self.csv_path)

        if csv_directory:
            os.makedirs(
                csv_directory,
                exist_ok=True
            )

    @staticmethod
    def _clean_number(value, decimals=6):
        value = float(value)

        if math.isnan(value):
            return ""

        return round(value, decimals)

    def append_experiment(
        self,
        experiment,
        model,
        activation,
        environment,
        ablation,
        training_timesteps,
        evaluation_episodes,
        mean_covered_area_m2,
        total_collisions,
        total_falls,
        mean_reward_start,
        mean_reward_end,
        reward_improvement
    ):
        file_exists = os.path.isfile(self.csv_path)

        row = {
            "experiment": str(experiment),
            "model": str(model),
            "activation": str(activation),
            "environment": str(environment),
            "ablation": str(ablation),
            "training_timesteps": int(training_timesteps),
            "evaluation_episodes": int(evaluation_episodes),
            "mean_covered_area_m2": self._clean_number(
                mean_covered_area_m2
            ),
            "total_collisions": int(total_collisions),
            "total_falls": int(total_falls),
            "mean_reward_start": self._clean_number(
                mean_reward_start
            ),
            "mean_reward_end": self._clean_number(
                mean_reward_end
            ),
            "reward_improvement": self._clean_number(
                reward_improvement
            )
        }

        with open(
            self.csv_path,
            mode="a",
            newline="",
            encoding="utf-8"
        ) as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=self.FIELDNAMES
            )

            if not file_exists:
                writer.writeheader()

            writer.writerow(row)
