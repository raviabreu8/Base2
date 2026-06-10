#
# ISCTE-IUL, IAR, 2024/2025.
#
# Template to use SB3 to train a Thymio in Webots.
#

import os
import sys

try:
    import gymnasium as gym
    import numpy as np
    import torch.nn as nn

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from sb3_contrib import RecurrentPPO
    from controller import Supervisor

    from thymio_sensors import ThymioSensors
    from thymio_motors import ThymioMotors
    from thymio_robot import ThymioRobot
    from thymio_episode import ThymioEpisode
    from thymio_safety import ThymioSafety
    from thymio_reward import ThymioReward
    from thymio_obstacles import ThymioObstacles
    from results import (
        SimpleResultsCSV,
        TrainingRewardCallback
    )

except ImportError:
    sys.exit(
        "Please make sure you have all dependencies installed."
    )


class OpenAIGymEnvironment(Supervisor, gym.Env):

    metadata = {
        "render_modes": []
    }

    def __init__(
        self,
        max_episode_steps=600,
        obstacle_mode="random"
    ):
        super().__init__()

        if obstacle_mode not in (
            "random",
            "standard"
        ):
            raise ValueError(
                "obstacle_mode deve ser 'random' ou 'standard'."
            )

        self.spec = gym.envs.registration.EnvSpec(
            id="WebotsEnv-v0",
            entry_point=(
                "thymio_rl_controller:OpenAIGymEnvironment"
            ),
            max_episode_steps=max_episode_steps
        )

        self.__timestep = int(
            self.getBasicTimeStep()
        )

        self.obstacle_mode = obstacle_mode

        ## Seed fixa usada apenas para o layout standard.
        self.standard_obstacle_seed = 12345

        ## Ação: [motor esquerdo, motor direito].
        self.action_space = gym.spaces.Box(
            low=np.array(
                [-1.0, -1.0],
                dtype=np.float32
            ),
            high=np.array(
                [1.0, 1.0],
                dtype=np.float32
            ),
            dtype=np.float32
        )

        ## Observação: 5 sensores frontais + 2 de chão.
        self.observation_space = gym.spaces.Box(
            low=np.zeros(
                7,
                dtype=np.float32
            ),
            high=np.ones(
                7,
                dtype=np.float32
            ),
            dtype=np.float32
        )

        self.state = None

        self.sensors = None
        self.motors = None
        self.robot_pose = None
        self.safety = None
        self.reward_function = None
        self.obstacles = None

        self.episode = ThymioEpisode(
            max_episode_steps=max_episode_steps
        )

        self.initial_yaw = 0.0

    def reset(
        self,
        seed=None,
        options=None
    ):
        gym.Env.reset(
            self,
            seed=seed
        )

        self.simulationReset()
        self.simulationResetPhysics()

        super().step(
            self.__timestep
        )

        self.sensors = ThymioSensors(
            robot=self,
            timestep=self.__timestep
        )

        self.motors = ThymioMotors(
            robot=self,
            speed_factor=0.50
        )

        self.robot_pose = ThymioRobot(
            supervisor=self
        )

        self.obstacles = ThymioObstacles(
            supervisor=self,
            floor_height=(
                self.robot_pose.initial_translation[2]
            ),
            number_of_obstacles=4
        )

        self.safety = ThymioSafety(
            robot_pose=self.robot_pose,
            cliff_warning_threshold=0.45,
            cliff_threshold=0.30,
            ground_change_threshold=0.30,
            cliff_warning_limit=2,
            fall_height_drop=0.025,
            fall_tilt_degrees=8.0
        )

        self.reward_function = ThymioReward(
            cell_size=0.20,
            survival_reward=0.05,
            new_cell_reward=0.10,
            revisit_penalty=-0.05,
            max_forward_reward=0.15,
            obstacle_penalty_scale=0.30,
            rotation_penalty_scale=0.02,
            cliff_warning_penalty=-1.00,
            cliff_penalty=-20.00,
            collision_penalty=-5.00,
            fall_penalty=-20.00
        )

        self.motors.stop()

        ## Random: obstáculos diferentes por episódio.
        ## Standard: o mesmo layout em todos os episódios.
        if self.obstacle_mode == "standard":
            obstacle_random_generator = (
                np.random.default_rng(
                    self.standard_obstacle_seed
                )
            )
        else:
            obstacle_random_generator = self.np_random

        obstacle_information = (
            self.obstacles.randomize(
                random_generator=(
                    obstacle_random_generator
                )
            )
        )

        ## A orientação inicial continua aleatória.
        self.initial_yaw = (
            self.robot_pose.reset_pose(
                random_generator=self.np_random
            )
        )

        self.episode.reset()

        for _ in range(15):
            super().step(
                self.__timestep
            )

        self.state = np.asarray(
            self.sensors.read_normalized(),
            dtype=np.float32
        )

        ## Guarda a primeira leitura dos sensores de chão
        ## para comparar com o passo seguinte.
        self.safety.reset(
            observation=self.state
        )

        initial_position = (
            self.robot_pose
            .robot_node
            .getPosition()
        )

        self.reward_function.reset(
            initial_position=initial_position
        )

        return (
            self.state.copy(),
            {
                "step_count": 0,
                "initial_yaw": self.initial_yaw,
                "unique_cells": 1,
                "obstacles": obstacle_information
            }
        )

    def step(
        self,
        action
    ):
        self.episode.advance()

        action = np.asarray(
            action,
            dtype=np.float32
        )

        action = np.clip(
            action,
            self.action_space.low,
            self.action_space.high
        )

        self.motors.apply_action(
            action
        )

        for _ in range(5):
            result = super().step(
                self.__timestep
            )

            if result == -1:
                break

        self.state = np.asarray(
            self.sensors.read_normalized(),
            dtype=np.float32
        )

        (
            critical_cliff,
            cliff_warning,
            confirmed_cliff,
            delta_gs,
            gs_min,
            gs_left,
            gs_right,
            cliff_warning_count
        ) = self.safety.detect_cliff(
            self.state
        )

        (
            fall,
            robot_height,
            height_drop,
            tilt_degrees
        ) = self.safety.detect_fall()

        ## Colisão aproximada: pelo menos um sensor frontal
        ## está praticamente saturado.
        max_front_proximity = float(
            np.max(
                self.state[:5]
            )
        )

        collision = bool(
            max_front_proximity
            >= 0.95
        )

        position = (
            self.robot_pose
            .robot_node
            .getPosition()
        )

        (
            reward,
            reward_info
        ) = self.reward_function.compute(
            action=action,
            observation=self.state,
            position=position,
            cliff_warning=cliff_warning,
            cliff_terminal=critical_cliff,
            collision=collision,
            fall=fall
        )

        terminated = bool(
            fall
            or critical_cliff
            or collision
        )

        truncated = bool(
            self.episode.is_truncated()
            and not terminated
        )

        if terminated or truncated:
            self.motors.stop()

        info = {
            "step_count": self.episode.step_count,
            "position": np.asarray(
                position,
                dtype=np.float32
            ),
            "cliff": bool(critical_cliff),
            "cliff_warning": bool(cliff_warning),
            "confirmed_cliff": bool(confirmed_cliff),
            "delta_gs": float(delta_gs),
            "gs_min": float(gs_min),
            "gs_left": float(gs_left),
            "gs_right": float(gs_right),
            "cliff_warning_count": int(
                cliff_warning_count
            ),
            "collision": bool(collision),
            "max_front_proximity": float(
                max_front_proximity
            ),
            "fall": bool(fall),
            "robot_height": float(robot_height),
            "height_drop": float(height_drop),
            "tilt_degrees": float(tilt_degrees),
            "termination_reason": (
                "fall"
                if fall
                else (
                    "cliff"
                    if critical_cliff
                    else (
                        "collision"
                        if collision
                        else None
                    )
                )
            ),
            "truncation_reason": (
                "time_limit"
                if truncated
                else None
            )
        }

        info.update(
            reward_info
        )

        return (
            self.state.copy(),
            float(reward),
            bool(terminated),
            bool(truncated),
            info
        )


def get_activation_class(
    activation_name
):
    if activation_name == "Tanh":
        return nn.Tanh

    if activation_name == "ReLU":
        return nn.ReLU

    raise ValueError(
        "activation_name deve ser 'Tanh' ou 'ReLU'."
    )


def create_model(
    model_type,
    activation_name,
    env
):
    policy_kwargs = {
        "activation_fn": get_activation_class(
            activation_name
        )
    }

    common_parameters = {
        "env": env,
        "learning_rate": 3e-4,
        "n_steps": 256,
        "batch_size": 64,
        "n_epochs": 5,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.001,
        "policy_kwargs": policy_kwargs,
        "verbose": 1,
        "seed": 42,
        "device": "cpu"
    }

    if model_type == "PPO":
        return PPO(
            policy="MlpPolicy",
            **common_parameters
        )

    if model_type == "RecurrentPPO":
        return RecurrentPPO(
            policy="MlpLstmPolicy",
            **common_parameters
        )

    raise ValueError(
        "model_type deve ser 'PPO' ou 'RecurrentPPO'."
    )


def load_model(
    model_type,
    model_path
):
    if model_type == "PPO":
        return PPO.load(
            model_path,
            device="cpu"
        )

    return RecurrentPPO.load(
        model_path,
        device="cpu"
    )



































def main():

    # =========================================================
    # CONFIGURAÇÃO DA EXPERIÊNCIA
    # =========================================================

    ## True: carrega o modelo existente e apenas avalia.
    ## False: treina um modelo novo e depois avalia.
    only_evaluate = True

    ## Altera estes cinco valores entre experiências.
    experiment_name = "ppo_tanh_random_cliff_warning_v3_1"
    model_type = "PPO"
    activation_name = "Tanh"
    environment_name = "random"
    ablation_name = "none"

    total_training_steps = 20480
    max_episode_steps = 600

    evaluation_seeds = [
        100,
        101,
        102,
        103,
        104
    ]

    env = OpenAIGymEnvironment(
        max_episode_steps=max_episode_steps,
        obstacle_mode=environment_name
    )

    controller_directory = os.path.dirname(
        os.path.abspath(__file__)
    )

    model_directory = os.path.join(
        controller_directory,
        "models"
    )

    checkpoint_directory = os.path.join(
        controller_directory,
        "checkpoints"
    )

    results_directory = os.path.join(
        controller_directory,
        "results"
    )

    os.makedirs(
        model_directory,
        exist_ok=True
    )

    os.makedirs(
        checkpoint_directory,
        exist_ok=True
    )

    final_model_path = os.path.join(
        model_directory,
        experiment_name
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=5120,
        save_path=checkpoint_directory,
        name_prefix=experiment_name
    )

    training_reward_callback = (
        TrainingRewardCallback()
    )

    results_csv = SimpleResultsCSV(
        csv_path=os.path.join(
            results_directory,
            "results.csv"
        )
    )

    ## Só cria uma rede nova quando vamos treinar.
    if only_evaluate:
        model = None
    else:
        model = create_model(
            model_type=model_type,
            activation_name=activation_name,
            env=env
        )

    try:
        # =====================================================
        # TREINO OU CARREGAMENTO
        # =====================================================

        if not only_evaluate:
            print()
            print("=" * 70)
            print("TREINO")
            print("=" * 70)
            print("Experiência:", experiment_name)
            print("Modelo:", model_type)
            print("Ativação:", activation_name)
            print("Ambiente:", environment_name)
            print("Timesteps:", total_training_steps)

            model.learn(
                total_timesteps=(
                    total_training_steps
                ),
                callback=[
                    checkpoint_callback,
                    training_reward_callback
                ],
                reset_num_timesteps=True,
                progress_bar=False
            )

            model.save(
                final_model_path
            )

            (
                mean_reward_start,
                mean_reward_end,
                reward_improvement
            ) = training_reward_callback.get_reward_summary(
                window_size=10
            )

            print()
            print("TREINO TERMINADO")
            print(
                "Reward média inicial:",
                round(mean_reward_start, 3)
            )
            print(
                "Reward média final:",
                round(mean_reward_end, 3)
            )
            print(
                "Melhoria:",
                round(reward_improvement, 3)
            )

        else:
            saved_model_file = (
                final_model_path
                + ".zip"
            )

            if not os.path.exists(
                saved_model_file
            ):
                raise FileNotFoundError(
                    "Modelo não encontrado: "
                    + saved_model_file
                )

            print()
            print("=" * 70)
            print("MODO APENAS AVALIAÇÃO")
            print("=" * 70)
            print(
                "Modelo carregado:",
                saved_model_file
            )

            mean_reward_start = float("nan")
            mean_reward_end = float("nan")
            reward_improvement = float("nan")

        # =====================================================
        # AVALIAÇÃO
        # =====================================================

        evaluation_model = load_model(
            model_type=model_type,
            model_path=final_model_path
        )

        covered_areas = []
        total_collisions = 0
        total_falls = 0
        total_cliff_terminations = 0

        cell_size = 0.20
        collision_threshold = 0.95

        print()
        print("=" * 70)
        print("AVALIAÇÃO")
        print("=" * 70)

        for episode_number, evaluation_seed in enumerate(
            evaluation_seeds,
            start=1
        ):
            observation, _ = env.reset(
                seed=evaluation_seed
            )

            terminated = False
            truncated = False

            episode_reward = 0.0
            episode_collisions = 0
            collision_active = False

            lstm_states = None
            episode_starts = np.ones(
                (1,),
                dtype=bool
            )

            while (
                not terminated
                and not truncated
            ):
                if model_type == "RecurrentPPO":
                    action, lstm_states = (
                        evaluation_model.predict(
                            observation,
                            state=lstm_states,
                            episode_start=(
                                episode_starts
                            ),
                            deterministic=True
                        )
                    )
                else:
                    action, _ = (
                        evaluation_model.predict(
                            observation,
                            deterministic=True
                        )
                    )

                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info
                ) = env.step(
                    action
                )

                episode_reward += float(
                    reward
                )

                obstacle_risk = float(
                    info["obstacle_risk"]
                )

                ## Conta apenas a entrada num evento de colisão.
                if (
                    obstacle_risk
                    >= collision_threshold
                ):
                    if not collision_active:
                        episode_collisions += 1

                    collision_active = True
                else:
                    collision_active = False

                episode_starts = np.array(
                    [terminated or truncated],
                    dtype=bool
                )

            unique_cells = int(
                info["unique_cells"]
            )

            covered_area_m2 = float(
                unique_cells
                * cell_size
                * cell_size
            )

            covered_areas.append(
                covered_area_m2
            )

            total_collisions += (
                episode_collisions
            )

            if info["fall"]:
                total_falls += 1

            episode_reason = (
                info["termination_reason"]
                if terminated
                else info["truncation_reason"]
            )

            if episode_reason == "cliff":
                total_cliff_terminations += 1

            print()
            print(
                "Episódio",
                episode_number,
                "| seed:",
                evaluation_seed
            )
            print(
                "  Reward:",
                round(episode_reward, 3)
            )
            print(
                "  Células:",
                unique_cells
            )
            print(
                "  Área:",
                round(covered_area_m2, 4),
                "m²"
            )
            print(
                "  Colisões aproximadas:",
                episode_collisions
            )
            print(
                "  Queda:",
                bool(info["fall"])
            )
            print(
                "  Motivo:",
                episode_reason
            )
            if episode_reason == "cliff":
                print(
                    "  Cliff confirmado:",
                    info["confirmed_cliff"]
                )

                print(
                    "  Menor sensor de chão:",
                    round(info["gs_min"], 3)
                )
                print(
                    "  Ground esquerdo:",
                     round(info["gs_left"], 3)
                )
                print(
                    "  Ground direito:",
                    round(info["gs_right"], 3)
                )

                print(
                    "  Contador de avisos:",
                    info["cliff_warning_count"]
                )

        mean_covered_area_m2 = float(
            np.mean(
                covered_areas
            )
        )

        # =====================================================
        # GUARDAR UMA ÚNICA LINHA NO results.csv
        # =====================================================

        ## Em modo apenas avaliação não cria uma linha duplicada.
        if not only_evaluate:
            results_csv.append_experiment(
                experiment=experiment_name,
                model=model_type,
                activation=activation_name,
                environment=environment_name,
                ablation=ablation_name,
                training_timesteps=(
                    total_training_steps
                ),
                evaluation_episodes=len(
                    evaluation_seeds
                ),
                mean_covered_area_m2=(
                    mean_covered_area_m2
                ),
                total_collisions=(
                    total_collisions
                ),
                total_falls=total_falls,
                total_cliff_terminations=(
                    total_cliff_terminations
                ),
                mean_reward_start=(
                    mean_reward_start
                ),
                mean_reward_end=(
                    mean_reward_end
                ),
                reward_improvement=(
                    reward_improvement
                )
            )

        print()
        print("=" * 70)
        print(
            "RESULTADO GUARDADO"
            if not only_evaluate
            else "RESUMO DA AVALIAÇÃO"
        )
        print("=" * 70)

        if not only_evaluate:
            print(
                "CSV:",
                results_csv.csv_path
            )

        print(
            "Área média coberta:",
            round(mean_covered_area_m2, 4),
            "m²"
        )
        print(
            "Total de colisões:",
            total_collisions
        )
        print(
            "Total de quedas:",
            total_falls
        )
        print(
            "Total de terminações por cliff:",
            total_cliff_terminations
        )

    finally:
        if env.motors is not None:
            env.motors.stop()

        print()
        print(
            "Treino e avaliação terminados. "
            "Motores parados."
        )


if __name__ == "__main__":
    main()
