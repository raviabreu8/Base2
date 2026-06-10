import math


class ThymioObstacles:
    """
    Cria obstáculos aleatórios diretamente no mundo Webots.

    Não é necessário adicionar previamente os obstáculos
    ao ficheiro .wbt.
    """

    def __init__(
        self,
        supervisor,
        floor_height,
        number_of_obstacles=4
    ):
        ## NOVO:
        ## Guarda o Supervisor para poder alterar o mundo.
        self.supervisor = supervisor

        ## NOVO:
        ## Altura da superfície onde o robô circula.
        self.floor_height = float(
            floor_height
        )

        ## NOVO:
        ## Número de obstáculos criados por episódio.
        self.number_of_obstacles = int(
            number_of_obstacles
        )

        ## NOVO:
        ## Campo children da raiz do mundo.
        ## É aqui que os novos obstáculos serão inseridos.
        self.world_children = (
            self.supervisor
            .getRoot()
            .getField("children")
        )

        if self.world_children is None:
            raise RuntimeError(
                "Não foi possível obter o campo "
                "children da raiz do mundo."
            )

        ## NOVO:
        ## Limites usados para posicionar as caixas.
        self.x_range = (
            -0.38,
            0.34
        )

        self.top_y_range = (
            0.38,
            0.62
        )

        self.bottom_y_range = (
            -0.62,
            -0.38
        )

        ## NOVO:
        ## Intervalos das dimensões.
        self.size_x_range = (
            0.06,
            0.12
        )

        self.size_y_range = (
            0.06,
            0.14
        )

        self.height_range = (
            0.10,
            0.18
        )

        ## NOVO:
        ## Distância mínima entre obstáculos.
        self.minimum_distance = 0.20

    def remove_existing(self):
        """
        Remove obstáculos criados anteriormente.
        """

        for index in range(
            self.number_of_obstacles
        ):
            obstacle_node = (
                self.supervisor.getFromDef(
                    "RANDOM_OBSTACLE_{}".format(
                        index
                    )
                )
            )

            if obstacle_node is not None:
                obstacle_node.remove()

    def _sample_position(
        self,
        random_generator,
        y_range,
        previous_positions
    ):
        """
        Sorteia uma posição que não esteja demasiado
        próxima das posições já escolhidas.
        """

        for _ in range(100):
            x = float(
                random_generator.uniform(
                    self.x_range[0],
                    self.x_range[1]
                )
            )

            y = float(
                random_generator.uniform(
                    y_range[0],
                    y_range[1]
                )
            )

            valid_position = True

            for previous_x, previous_y in (
                previous_positions
            ):
                distance = math.hypot(
                    x - previous_x,
                    y - previous_y
                )

                if (
                    distance
                    < self.minimum_distance
                ):
                    valid_position = False
                    break

            if valid_position:
                return x, y

        raise RuntimeError(
            "Não foi possível encontrar "
            "uma posição válida para o obstáculo."
        )

    def randomize(self, random_generator):
        """
        Remove os obstáculos antigos e cria
        um novo conjunto aleatório.
        """

        ## NOVO:
        ## Limpa possíveis obstáculos anteriores.
        self.remove_existing()

        obstacle_information = []
        previous_positions = []

        for index in range(
            self.number_of_obstacles
        ):
            ## NOVO:
            ## Alterna obstáculos entre a zona
            ## superior e inferior do ambiente.
            if index % 2 == 0:
                y_range = (
                    self.top_y_range
                )
            else:
                y_range = (
                    self.bottom_y_range
                )

            x, y = self._sample_position(
                random_generator=(
                    random_generator
                ),
                y_range=y_range,
                previous_positions=(
                    previous_positions
                )
            )

            previous_positions.append(
                (x, y)
            )

            ## NOVO:
            ## Dimensões aleatórias.
            size_x = float(
                random_generator.uniform(
                    self.size_x_range[0],
                    self.size_x_range[1]
                )
            )

            size_y = float(
                random_generator.uniform(
                    self.size_y_range[0],
                    self.size_y_range[1]
                )
            )

            height = float(
                random_generator.uniform(
                    self.height_range[0],
                    self.height_range[1]
                )
            )

            ## NOVO:
            ## Orientação aleatória.
            yaw = float(
                random_generator.uniform(
                    -math.pi,
                    math.pi
                )
            )

            ## NOVO:
            ## Coloca a base da caixa sobre a plataforma.
            z = (
                self.floor_height
                + height / 2.0
            )

            ## NOVO:
            ## Texto que define um novo SolidBox no Webots.
            obstacle_definition = """
DEF RANDOM_OBSTACLE_{0} SolidBox {{
  translation {1:.6f} {2:.6f} {3:.6f}
  rotation 0 0 1 {4:.6f}
  name "random obstacle {0}"
  size {5:.6f} {6:.6f} {7:.6f}
  appearance PBRAppearance {{
    baseColor 0.8 0.2 0.2
    roughness 0.5
    metalness 0
  }}
}}
""".format(
                index,
                x,
                y,
                z,
                yaw,
                size_x,
                size_y,
                height
            )

            ## NOVO:
            ## Cria efetivamente o obstáculo no mundo.
            self.world_children.importMFNodeFromString(
                -1,
                obstacle_definition
            )

            obstacle_information.append(
                {
                    "index": index,
                    "position": [
                        x,
                        y,
                        z
                    ],
                    "size": [
                        size_x,
                        size_y,
                        height
                    ],
                    "yaw": yaw
                }
            )

        return obstacle_information

