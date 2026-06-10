import math


class ThymioRobot:
    """
    Gere o nó do robô e a sua pose inicial no Webots.
    """

    def __init__(self, supervisor):
        ## NOVO:
        ## Obtém o nó do Thymio definido no mundo como DEF ROBOT.
        self.robot_node = supervisor.getFromDef(
            "ROBOT"
        )

        if self.robot_node is None:
            raise RuntimeError(
                "Não foi encontrado um nó DEF ROBOT no mundo."
            )

        ## NOVO:
        ## Obtém o campo que permite alterar a posição (XYZ)
        self.translation_field = (
            self.robot_node.getField(
                "translation"
            )
        )
        ## Obtém os campo que permite alterar a orientação
        self.rotation_field = (
            self.robot_node.getField(
                "rotation"
            )
        )

        ## NOVO:
        ## Guarda a posição inicial definida no ficheiro .wbt.
        ## Esta posição corresponde ao centro do ambiente.
        self.initial_translation = (
            self.translation_field.getSFVec3f()
        )

    def reset_pose(self, random_generator):
        """
        Coloca o robô no centro com orientação aleatória.
        """

        ## NOVO:
        ## Escolhe uma rotação aleatória entre -pi e pi.
        random_yaw = float(
            random_generator.uniform(
                -math.pi,
                math.pi
            )
        )

        ## NOVO:
        ## Recoloca o robô na posição inicial do mundo.
        self.translation_field.setSFVec3f(
            self.initial_translation
        )

        ## NOVO:
        ## Define uma rotação em torno do eixo vertical Z.
        self.rotation_field.setSFRotation(
            [
                0.0,
                0.0,
                1.0,
                random_yaw
            ]
        )

        ## NOVO:
        ## Elimina velocidades e forças físicas anteriores.
        self.robot_node.resetPhysics()

        return random_yaw
