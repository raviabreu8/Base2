import math
import numpy as np


class ThymioSafety:
    """
    Deteta avisos de precipício, precipícios críticos
    e quedas do Thymio.
    """

    def __init__(
        self,
        robot_pose,
        cliff_warning_threshold=0.45,
        cliff_threshold=0.30,
        ground_change_threshold=0.30,
        cliff_warning_limit=2,
        fall_height_drop=0.025,
        fall_tilt_degrees=8.0
    ):
        self.robot_node = (
            robot_pose.robot_node
        )

        self.initial_height = float(
            robot_pose.initial_translation[2]
        )

        ## Abaixo deste valor começa a zona de aviso.
        self.cliff_warning_threshold = float(
            cliff_warning_threshold
        )

        ## Abaixo deste valor o precipício é crítico.
        self.cliff_threshold = float(
            cliff_threshold
        )

        ## Mudança mínima da leitura para gerar aviso.
        self.ground_change_threshold = float(
            ground_change_threshold
        )

        ## Número de avisos acumulados para tornar o cliff terminal.
        self.cliff_warning_limit = int(
            cliff_warning_limit
        )

        self.fall_height_drop = float(
            fall_height_drop
        )

        self.fall_tilt_degrees = float(
            fall_tilt_degrees
        )

        ## Menor leitura dos sensores no passo anterior.
        self.previous_gs_min = None

        ## Número de avisos acumulados.
        self.cliff_warning_count = 0

    def reset(
        self,
        observation
    ):
        """Reinicia a memória dos sensores no início do episódio."""

        ground_sensors = np.asarray(
            observation[5:7],
            dtype=np.float32
        )

        self.previous_gs_min = float(
            np.min(
                ground_sensors
            )
        )

        self.cliff_warning_count = 0

    def detect_cliff(
        self,
        observation
    ):
        """
        Deteta aproximação e risco crítico de precipício.

        Devolve:
            critical_cliff,
            cliff_warning,
            confirmed_cliff,
            delta_gs,
            gs_min,
            cliff_warning_count
        """

        ground_sensors = np.asarray(
            observation[5:7],
            dtype=np.float32
        )

        gs_left = float(
            ground_sensors[0]
        )

        gs_right = float(
            ground_sensors[1]
        )

        ## O sensor com menor leitura é o mais próximo da borda.
        gs_min = float(
            min(
                gs_left,
                gs_right
            )
        )

        if self.previous_gs_min is None:
            delta_gs = 0.0
        else:
            delta_gs = float(
                abs(
                    gs_min
                    - self.previous_gs_min
                )
            )

        ## Nível 1: aproximação à borda ou mudança rápida.
        warning_signal = bool(
            gs_min
            < self.cliff_warning_threshold
            or
            delta_gs
            > self.ground_change_threshold
        )

        if warning_signal:
            self.cliff_warning_count += 1
        else:
            ## Em chão seguro, o aviso diminui gradualmente.
            self.cliff_warning_count = max(
                0,
                self.cliff_warning_count - 1
            )

        cliff_warning = bool(
            self.cliff_warning_count > 0
        )

        ## Nível crítico: leitura já muito baixa.
        #apenas diagnostico n entra na recompensa
        confirmed_cliff = bool(
            gs_min
            < self.cliff_threshold
        )

        ## Termina por leitura crítica ou dois avisos acumulados.
        critical_cliff = bool(
            self.cliff_warning_count
            >= self.cliff_warning_limit
        )

        self.previous_gs_min = gs_min

        return (
            critical_cliff,
            cliff_warning,
            confirmed_cliff,
            delta_gs,
            gs_min,
            gs_left,
            gs_right,
            self.cliff_warning_count
        )

    def get_tilt_degrees(self):
        """Calcula a inclinação do robô relativamente à vertical."""

        orientation = (
            self.robot_node.getOrientation()
        )

        vertical_alignment = float(
            np.clip(
                orientation[8],
                -1.0,
                1.0
            )
        )

        return math.degrees(
            math.acos(
                vertical_alignment
            )
        )

    def detect_fall(self):
        """Deteta uma queda pela altura ou inclinação."""

        position = (
            self.robot_node.getPosition()
        )

        robot_height = float(
            position[2]
        )

        height_drop = (
            self.initial_height
            - robot_height
        )

        tilt_degrees = (
            self.get_tilt_degrees()
        )

        fall = bool(
            height_drop
            > self.fall_height_drop
            or
            tilt_degrees
            > self.fall_tilt_degrees
        )

        return (
            fall,
            robot_height,
            height_drop,
            tilt_degrees
        )
