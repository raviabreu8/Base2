import numpy as np


class ThymioSensors:
    """
    Gere os 5 sensores frontais e os 2 sensores de chão do Thymio.
    """

    FRONT_SENSOR_MAX = 4308.0
    GROUND_SENSOR_MAX = 1000.0

    def __init__(self, robot, timestep):
        self.front_sensors = [
            robot.getDevice(f"prox.horizontal.{index}")
            for index in range(5)
        ]

        self.ground_sensors = [
            robot.getDevice("prox.ground.0"),
            robot.getDevice("prox.ground.1"),
        ]

        for sensor in self.front_sensors + self.ground_sensors:
            sensor.enable(timestep)

    def read_raw(self):
        """
        Lê os valores brutos dos sensores.
        """

        front_raw = np.array(
            [
                sensor.getValue()
                for sensor in self.front_sensors
            ],
            dtype=np.float32,
        )

        ground_raw = np.array(
            [
                sensor.getValue()
                for sensor in self.ground_sensors
            ],
            dtype=np.float32,
        )

        return front_raw, ground_raw

    def read_normalized(self):
        """
        Devolve os 7 sensores normalizados para [0, 1].

        Ordem:
        [front_0, front_1, front_2, front_3, front_4,
         ground_0, ground_1]
        """

        front_raw, ground_raw = self.read_raw()

        front_normalized = np.clip(
            front_raw / self.FRONT_SENSOR_MAX,
            0.0,
            1.0,
        )

        ground_normalized = np.clip(
            ground_raw / self.GROUND_SENSOR_MAX,
            0.0,
            1.0,
        )

        observation = np.concatenate(
            [
                front_normalized,
                ground_normalized,
            ]
        )

        return observation.astype(np.float32)