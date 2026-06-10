class ThymioMotors:
    """
    Gere os dois motores do Thymio.

    A ação recebida tem o formato:
        [velocidade_esquerda, velocidade_direita]

    Cada valor da ação deve estar entre -1 e 1.
    """

    def __init__(
        self,
        robot,
        speed_factor=0.50
    ):
        self.left_motor = robot.getDevice(
            "motor.left"
        )

        self.right_motor = robot.getDevice(
            "motor.right"
        )

        # Permite controlar diretamente a velocidade.
        self.left_motor.setPosition(
            float("inf")
        )

        self.right_motor.setPosition(
            float("inf")
        )

        physical_max_speed = min(
            self.left_motor.getMaxVelocity(),
            self.right_motor.getMaxVelocity()
        )

        # Usa apenas uma fração da velocidade física máxima.
        self.max_speed = (
            physical_max_speed
            * speed_factor
        )

        self.stop()

    def apply_action(self, action):
        """
        Converte uma ação normalizada em velocidades físicas.
        """

        left_action = float(action[0])
        right_action = float(action[1])

        # Proteção para manter os valores entre -1 e 1.
        left_action = max(
            -1.0,
            min(1.0, left_action)
        )

        right_action = max(
            -1.0,
            min(1.0, right_action)
        )

        left_velocity = (
            left_action
            * self.max_speed
        )

        right_velocity = (
            right_action
            * self.max_speed
        )

        self.left_motor.setVelocity(
            left_velocity
        )

        self.right_motor.setVelocity(
            right_velocity
        )

    def stop(self):
        """
        Para os dois motores.
        """

        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
