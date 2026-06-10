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
    - penalizações terminais  colisão e queda.
    """

    def __init__(
        self,
        cell_size=0.20,
        survival_reward=0.01,
        new_cell_reward=0.10,
        revisit_penalty=-0.05,
        max_forward_reward=0.15,
        obstacle_penalty_scale=0.30,
        rotation_penalty_scale=0.02,
        cliff_warning_penalty=-1.00,
        ground_recovery_scale=2.00,
        dangerous_forward_penalty_scale=0.50,
        reverse_recovery_scale=1.00,
        turn_away_scale=1.00,
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

        ## Recompensa quando o sensor de chão melhora,
        ## indicando recuperação da borda.
        self.ground_recovery_scale = float(
            ground_recovery_scale
        )

        ## Penalização por tentar continuar em frente
        ## enquanto existe risco de cliff.
        self.dangerous_forward_penalty_scale = float(
            dangerous_forward_penalty_scale
        )

        ## Recompensa imediata por recuar quando existe
        ## risco de precipício.
        self.reverse_recovery_scale = float(
            reverse_recovery_scale
        )

        ## Recompensa imediata por virar para o lado
        ## oposto ao sensor de chão mais baixo.
        self.turn_away_scale = float(
            turn_away_scale
        )

        ## Penalizações terminais.

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

        ## Memória interna usada para recompensar
        ## recuperação dos sensores de chão.
        self.previous_ground_sensor_min = None

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

        ## Memória interna usada para recompensar
        ## recuperação dos sensores de chão.
        self.previous_ground_sensor_min = None

        initial_cell = self.position_to_cell(
            initial_position
        )

        self.current_cell = initial_cell
        self.visited_cells.add(
            initial_cell
        )

        self.previous_ground_sensor_min = None

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

        ## Nesta estratégia, o cliff não termina o episódio.
        ## Apenas uma queda real ou colisão são eventos terminais.
        terminal_event = bool(
            collision
            or fall
        )

        ## Risco contínuo de precipício calculado diretamente
        ## a partir do menor dos dois sensores de chão.
        ground_sensors = observation[5:7]

        gs_left = float(
            ground_sensors[0]
        )

        gs_right = float(
            ground_sensors[1]
        )

        gs_min = float(
            min(
                gs_left,
                gs_right
            )
        )

        cliff_risk = float(
            np.clip(
                (
                    0.45
                    - gs_min
                )
                / 0.45,
                0.0,
                1.0
            )
        )

        if self.previous_ground_sensor_min is None:
            ground_improvement = 0.0
        else:
            ground_improvement = float(
                max(
                    gs_min
                    - self.previous_ground_sensor_min,
                    0.0
                )
            )

        ## Só dá bónus de recuperação se o robô
        ## estava ou está numa zona de risco.
        recovery_context = bool(
            (
                self.previous_ground_sensor_min
                is not None
                and self.previous_ground_sensor_min < 0.45
            )
            or gs_min < 0.45
        )

        ground_recovery_reward = float(
            self.ground_recovery_scale
            * ground_improvement
            if recovery_context
            and not terminal_event
            else 0.0
        )

        self.previous_ground_sensor_min = gs_min

        cliff_active = bool(
            cliff_risk > 0.0
        )

        ## Um aviso não termina, mas o passo já não é seguro.
        unsafe_event = bool(
            cliff_active
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

        ## Mede a diferença entre as rodas.
        ## 0 significa rodas iguais e 1 rotação máxima.
        rotation_amount = float(
            abs(
                float(action[0])
                - float(action[1])
            )
            / 2.0
        )

        ## Curvas suaves continuam a receber boa recompensa.
        ## Apenas rotações muito fortes reduzem bastante o avanço.
        ##
        ## Durante risco de cliff, avançar não recebe recompensa:
        ## o robô deve aprender a recuar ou virar para recuperar.
        forward_reward = float(
            self.max_forward_reward
            * max(
                linear_velocity,
                0.0
            )
            * (
                1.0
                - 0.50 * rotation_amount
            )
            if not cliff_active
            else 0.0
        )

        ## Se há risco de cliff, continuar a avançar
        ## para a frente é perigoso. Recuar não é penalizado.
        dangerous_forward_reward = float(
            -self.dangerous_forward_penalty_scale
            * max(
                linear_velocity,
                0.0
            )
            * cliff_risk
            if cliff_active
            and not terminal_event
            else 0.0
        )

        ## Em risco de precipício, recuar é imediatamente
        ## melhor do que continuar a avançar.
        reverse_recovery_reward = float(
            self.reverse_recovery_scale
            * max(
                -linear_velocity,
                0.0
            )
            * cliff_risk
            if cliff_active
            and not terminal_event
            else 0.0
        )

        ## Diferença entre os sensores: quanto maior, mais claro
        ## é de que lado está o precipício.
        ground_side_difference = float(
            abs(
                gs_left
                - gs_right
            )
        )

        turn_direction_confidence = float(
            np.clip(
                ground_side_difference
                / 0.45,
                0.0,
                1.0
            )
        )

        turn_away_amount = 0.0

        if cliff_active and not terminal_event:
            if gs_left < gs_right:
                ## Precipício à esquerda: roda esquerda mais
                ## rápida ajuda a virar para a direita.
                turn_away_amount = max(
                    (
                        float(action[0])
                        - float(action[1])
                    )
                    / 2.0,
                    0.0
                )

            elif gs_right < gs_left:
                ## Precipício à direita: roda direita mais
                ## rápida ajuda a virar para a esquerda.
                turn_away_amount = max(
                    (
                        float(action[1])
                        - float(action[0])
                    )
                    / 2.0,
                    0.0
                )

        turn_away_reward = float(
            self.turn_away_scale
            * turn_away_amount
            * cliff_risk
            * turn_direction_confidence
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

        rotation_reward = float(
            -self.rotation_penalty_scale
            * rotation_amount
        )

        # =====================================================
        # 6. AVISO DE PRECIPÍCIO
        # =====================================================

        ## Penalização contínua:
        ## 0.00 quando gs_min >= 0.45;
        ## aproxima-se de -1.00 quando gs_min chega a 0.
        ## O episódio continua para permitir recuperação.
        cliff_warning_reward = float(
            self.cliff_warning_penalty
            * cliff_risk
            if not terminal_event
            else 0.0
        )

        # =====================================================
        # 7. EVENTO TERMINAL
        # =====================================================

        if fall:
            terminal_reward = self.fall_penalty
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
            + ground_recovery_reward
            + dangerous_forward_reward
            + reverse_recovery_reward
            + turn_away_reward
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
            "reward_ground_recovery": ground_recovery_reward,
            "reward_dangerous_forward": dangerous_forward_reward,
            "reward_reverse_recovery": reverse_recovery_reward,
            "reward_turn_away": turn_away_reward,
            "turn_away_amount": float(turn_away_amount),
            "turn_direction_confidence": turn_direction_confidence,
            "ground_side_difference": ground_side_difference,
            "reward_terminal": terminal_reward,
            ## O cliff deixa de ter penalização terminal.
            ## Esta componente contém apenas o gradiente contínuo.
            "reward_cliff": cliff_warning_reward,
            "cliff_risk": cliff_risk,
            "ground_sensor_min": gs_min,
            "ground_sensor_left": gs_left,
            "ground_sensor_right": gs_right,
            "ground_improvement": ground_improvement,
            "reward_collision": (
                self.collision_penalty
                if collision
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
