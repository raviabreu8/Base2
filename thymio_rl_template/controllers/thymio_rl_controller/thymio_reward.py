import math
import numpy as np


class ThymioReward:
    """
    Função de recompensa baseada em:
    - sobrevivência;
    - avanço;
    - exploração de células novas;
    - penalização de revisitas;
    - penalização de obstáculos;
    - penalização de rotação excessiva;
    - penalizações terminais por precipício, colisão e queda.
    """

    def __init__(
        self,
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
    ):
        ## Tamanho de cada célula da grelha de exploração.
        self.cell_size = float(
            cell_size
        )

        ## Pequena recompensa por continuar vivo e sem evento terminal.
        self.survival_reward_value = float(
            survival_reward
        )

        ## Recompensa por entrar numa célula ainda não visitada.
        self.new_cell_reward = float(
            new_cell_reward
        )

        ## Penalização por voltar a uma célula já visitada.
        self.revisit_penalty = float(
            revisit_penalty
        )

        ## Recompensa máxima quando as duas rodas avançam no máximo.
        self.max_forward_reward = float(
            max_forward_reward
        )

        ## Escala da penalização quadrática de obstáculos.
        self.obstacle_penalty_scale = float(
            obstacle_penalty_scale
        )

        ## Escala da penalização por diferença entre as rodas.
        self.rotation_penalty_scale = float(
            rotation_penalty_scale
        )

        ## Penalização de aviso recuperável de precipício.
        self.cliff_warning_penalty = float(
            cliff_warning_penalty
        )

        ## Penalizações terminais.
        self.cliff_penalty = float(
            cliff_penalty
        )

        self.collision_penalty = float(
            collision_penalty
        )

        self.fall_penalty = float(
            fall_penalty
        )

        self.visited_cells = set()
        self.current_cell = None

        self.new_cell_count = 0
        self.revisit_count = 0

    def position_to_cell(
        self,
        position
    ):
        """Converte a posição XY numa célula da grelha."""

        x = float(
            position[0]
        )

        y = float(
            position[1]
        )

        return (
            int(
                math.floor(
                    x / self.cell_size
                )
            ),
            int(
                math.floor(
                    y / self.cell_size
                )
            )
        )

    def reset(
        self,
        initial_position
    ):
        """Reinicia a memória de exploração no início do episódio."""

        self.visited_cells = set()
        self.new_cell_count = 0
        self.revisit_count = 0

        initial_cell = self.position_to_cell(
            initial_position
        )

        self.current_cell = initial_cell
        self.visited_cells.add(
            initial_cell
        )

    def compute_exploration_reward(
        self,
        position,
        allow_update=True
    ):
        """Recompensa células novas e penaliza reentradas."""

        if not allow_update:
            return 0.0, False, False

        new_cell = self.position_to_cell(
            position
        )

        ## Ficar dentro da mesma célula não é uma revisita.
        if new_cell == self.current_cell:
            return 0.0, False, False

        self.current_cell = new_cell

        if new_cell not in self.visited_cells:
            self.visited_cells.add(
                new_cell
            )

            self.new_cell_count += 1

            return (
                self.new_cell_reward,
                True,
                False
            )

        self.revisit_count += 1

        return (
            self.revisit_penalty,
            False,
            True
        )

    def compute(
        self,
        action,
        observation,
        position,
        cliff_warning,
        cliff_terminal,
        collision,
        fall
    ):
        """Calcula a recompensa total de uma decisão do agente."""

        action = np.asarray(
            action,
            dtype=np.float32
        )

        observation = np.asarray(
            observation,
            dtype=np.float32
        )

        terminal_event = bool(
            cliff_terminal
            or collision
            or fall
        )

        ## Um aviso não termina, mas o passo já não é seguro.
        unsafe_event = bool(
            cliff_warning
            or terminal_event
        )

        # =====================================================
        # 1. SOBREVIVÊNCIA
        # =====================================================

        survival_reward = float(
            self.survival_reward_value
            if not unsafe_event
            else 0.0
        )

        # =====================================================
        # 2. EXPLORAÇÃO E REVISITAS
        # =====================================================

        (
            exploration_reward,
            entered_new_cell,
            revisited_cell
        ) = self.compute_exploration_reward(
            position=position,
            allow_update=(
                not unsafe_event
            )
        )

        # =====================================================
        # 3. MOVIMENTO PARA A FRENTE
        # =====================================================

        linear_velocity = float(
            (
                action[0]
                + action[1]
            )
            / 2.0
        )

        ## Só velocidades positivas recebem recompensa.
        forward_reward = float(
            self.max_forward_reward
            * max(
                linear_velocity,
                0.0
            )
        )

        # =====================================================
        # 4. OBSTÁCULOS
        # =====================================================

        front_sensors = observation[:5]

        obstacle_risk = float(
            np.max(
                front_sensors
            )
        )

        ## Penalização quadrática: pequena longe, forte muito perto.
        obstacle_reward = float(
            -self.obstacle_penalty_scale
            * (
                obstacle_risk ** 2
            )
        )

        # =====================================================
        # 5. ROTAÇÃO EXCESSIVA
        # =====================================================

        ## A diferença máxima possível entre as rodas é 2.
        rotation_amount = float(
            abs(
                float(action[0])
                - float(action[1])
            )
            / 2.0
        )

        rotation_reward = float(
            -self.rotation_penalty_scale
            * rotation_amount
        )

        # =====================================================
        # 6. AVISO DE PRECIPÍCIO
        # =====================================================

        ## Penalização leve: o episódio continua.
        cliff_warning_reward = float(
            self.cliff_warning_penalty
            if cliff_warning
            and not terminal_event
            else 0.0
        )

        # =====================================================
        # 7. EVENTO TERMINAL
        # =====================================================

        if fall:
            terminal_reward = self.fall_penalty
        elif cliff_terminal:
            terminal_reward = self.cliff_penalty
        elif collision:
            terminal_reward = self.collision_penalty
        else:
            terminal_reward = 0.0

        terminal_reward = float(
            terminal_reward
        )

        # =====================================================
        # REWARD TOTAL
        # =====================================================

        total_reward = float(
            survival_reward
            + exploration_reward
            + forward_reward
            + obstacle_reward
            + rotation_reward
            + cliff_warning_reward
            + terminal_reward
        )

        reward_info = {
            "reward_total": total_reward,
            "reward_survival": survival_reward,
            "reward_exploration": exploration_reward,
            "reward_forward": forward_reward,
            "reward_obstacle": obstacle_reward,
            "reward_rotation": rotation_reward,
            "reward_cliff_warning": cliff_warning_reward,
            "reward_terminal": terminal_reward,
            "reward_cliff": float(
                cliff_warning_reward
                + (
                    self.cliff_penalty
                    if cliff_terminal
                    and not fall
                    else 0.0
                )
            ),
            "reward_collision": (
                self.collision_penalty
                if collision
                and not cliff_terminal
                and not fall
                else 0.0
            ),
            "reward_fall": (
                self.fall_penalty
                if fall
                else 0.0
            ),
            "linear_velocity": linear_velocity,
            "rotation_amount": rotation_amount,
            "obstacle_risk": obstacle_risk,
            "entered_new_cell": entered_new_cell,
            "revisited_cell": revisited_cell,
            "current_cell": self.current_cell,
            "unique_cells": len(
                self.visited_cells
            ),
            "new_cell_count": self.new_cell_count,
            "revisit_count": self.revisit_count
        }

        return (
            total_reward,
            reward_info
        )
