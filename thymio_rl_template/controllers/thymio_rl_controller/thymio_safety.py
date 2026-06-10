import math
import numpy as np


class ThymioSafety:
    """
    Deteta precipícios e quedas do Thymio.
    """

    def __init__(
        self,
        robot_pose,
        cliff_threshold=0.30,
        ground_change_threshold=0.40,
        fall_height_drop=0.025,
        fall_tilt_degrees=8.0
    ):
        self.robot_node = (
            robot_pose.robot_node
        )

        self.initial_height = float(
            robot_pose.initial_translation[2]
        )

        ## Sensor de chão abaixo deste valor indica borda.
        self.cliff_threshold = float(
            cliff_threshold
        )

        ## Mudança súbita dos sensores de chão.
        self.ground_change_threshold = float(
            ground_change_threshold
        )

        self.fall_height_drop = float(
            fall_height_drop
        )

        self.fall_tilt_degrees = float(
            fall_tilt_degrees
        )

        ## Guarda a leitura anterior dos dois sensores de chão.
        self.previous_ground_sensors = None

    def reset(
        self,
        observation
    ):
        """Inicializa a memória dos sensores no começo do episódio."""

        self.previous_ground_sensors = np.asarray(
            observation[5:7],
            dtype=np.float32
        ).copy()

    def detect_cliff(
        self,
        observation
    ):
        """
        Deteta borda confirmada ou mudança súbita nos sensores de chão.

        Devolve:
            cliff_terminal,
            confirmed_cliff,
            sudden_ground_change,
            maximum_ground_change
        """

        current_ground_sensors = np.asarray(
            observation[5:7],
            dtype=np.float32
        )

        confirmed_cliff = bool(
            np.any(
                current_ground_sensors
                < self.cliff_threshold
            )
        )

        if self.previous_ground_sensors is None:
            maximum_ground_change = 0.0
        else:
            maximum_ground_change = float(
                np.max(
                    np.abs(
                        current_ground_sensors
                        - self.previous_ground_sensors
                    )
                )
            )

        sudden_ground_change = bool(
            maximum_ground_change
            > self.ground_change_threshold
        )

        cliff_terminal = bool(
            confirmed_cliff
            or sudden_ground_change
        )

        self.previous_ground_sensors = (
            current_ground_sensors.copy()
        )

        return (
            cliff_terminal,
            confirmed_cliff,
            sudden_ground_change,
            maximum_ground_change
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
