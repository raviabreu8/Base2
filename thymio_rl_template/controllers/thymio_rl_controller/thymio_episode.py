class ThymioEpisode:
    """
    Gere o número de passos de cada episódio.
    """

    def __init__(self, max_episode_steps):
        ## NOVO:
        ## Número máximo de decisões permitidas num episódio.
        self.max_episode_steps = int(
            max_episode_steps
        )

        ## NOVO:
        ## Contador de passos do episódio atual.
        self.step_count = 0

    def reset(self):
        """
        Reinicia o contador quando começa um novo episódio.
        """

        self.step_count = 0

    def advance(self):
        """
        Regista uma nova decisão executada pelo agente.
        """

        self.step_count += 1

    def is_truncated(self):
        """
        Indica se o episódio atingiu o limite de passos.
        """

        return (
            self.step_count
            >= self.max_episode_steps
        )
