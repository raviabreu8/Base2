README PARA APOIO AO RELATORIO
==============================

Este ficheiro resume a implementacao feita no controller do Thymio, as classes
principais, as experiencias treinadas e as conclusoes mais importantes para o
relatorio.


1. ESTRUTURA GERAL DO CONTROLLER
--------------------------------

O projeto foi dividido em varios ficheiros para separar responsabilidades:

- thymio_rl_controller.py
- thymio_sensors.py
- thymio_motors.py
- thymio_robot.py
- thymio_episode.py
- thymio_safety.py
- thymio_reward.py
- thymio_obstacles.py
- results.py

O ficheiro principal e o thymio_rl_controller.py. Ele cria o ambiente Gymnasium,
controla o Webots, treina/carrega modelos com Stable-Baselines3 e executa a
avaliacao final.


2. CLASSES E MODULOS PRINCIPAIS
-------------------------------

OpenAIGymEnvironment - thymio_rl_controller.py
Esta e a classe principal do ambiente de treino. Herda de Supervisor do Webots
e de gym.Env. Define:

- action_space: duas velocidades continuas, uma para cada roda, entre -1 e 1;
- observation_space: 7 sensores normalizados, sendo 5 frontais e 2 de chao;
- reset(): reinicia simulacao, robo, obstaculos, sensores e reward;
- step(): aplica acao, avanca a simulacao, calcula sensores, reward e condicoes
  de fim de episodio.

Tambem gere a diferenca entre ambiente "random" e "standard":

- random: obstaculos variam a cada episodio;
- standard: layout fixo de obstaculos, usado para comparar generalizacao.


ThymioSensors - thymio_sensors.py
Responsavel por ler e normalizar os sensores do robo. Usa:

- 5 sensores frontais: prox.horizontal.0 a prox.horizontal.4;
- 2 sensores de chao: prox.ground.0 e prox.ground.1.

A observacao final tem a forma:

[front_0, front_1, front_2, front_3, front_4, ground_0, ground_1]

Todos os valores sao normalizados para o intervalo [0, 1].


ThymioMotors - thymio_motors.py
Converte a acao produzida pelo modelo em velocidades fisicas dos motores. A acao
tem dois valores:

[motor_esquerdo, motor_direito]

Cada valor esta entre -1 e 1. Foi usado um speed_factor de 0.50 para nao usar a
velocidade fisica maxima do robo, tornando o comportamento mais controlavel.


ThymioRobot - thymio_robot.py
Gere a pose inicial do Thymio. Em cada episodio:

- o robo volta para a posicao inicial/central do mundo;
- a orientacao inicial e sorteada aleatoriamente entre -pi e pi;
- a fisica do robo e reiniciada.

Isto ajuda a testar o agente em condicoes iniciais diferentes.


ThymioEpisode - thymio_episode.py
Classe simples que conta os passos do episodio. O limite usado nas experiencias
principais foi:

max_episode_steps = 600

Quando este limite e atingido sem queda, colisao ou cliff/stuck, o episodio
termina por time_limit.


ThymioSafety - thymio_safety.py
Responsavel pela deteccao de risco e falhas de seguranca. Calcula:

- cliff_warning: aviso de aproximacao ao precipicio;
- confirmed_cliff: quando os dois sensores de chao estao abaixo do limiar;
- fall: queda real, usando perda de altura ou inclinacao;
- gs_min, gs_left, gs_right: valores dos sensores de chao;
- tilt_degrees e height_drop: usados para detetar queda.

Nota importante: o confirmed_cliff nao apanha todos os casos fisicos. Quando o
robo anda de costas ou fica com apenas uma roda fora, os sensores de chao podem
nao confirmar cliff, embora o estado seja perigoso.


Logica de stuck/cliff - thymio_rl_controller.py
Alem do ThymioSafety, foi adicionada uma logica de stuck no controller. A ideia
e terminar o episodio se o robo estiver em risco de cliff/inclinacao, estiver a
tentar mover-se e quase nao progredir.

A condicao usa:

- edge_or_tilt_risk;
- commanded_motion;
- low_progress;
- cliff_stuck_count.

Limitacao observada: se o robo ainda se move muito pouco, mesmo estando preso
com uma roda fora, o contador de stuck pode nao subir o suficiente. Isto foi
observado em alguns episodios.


ThymioReward - thymio_reward.py
Implementa a funcao de recompensa. A reward final combina:

- sobrevivencia;
- exploracao por celulas novas;
- penalizacao por revisitar celulas;
- pequeno incentivo a movimento para a frente;
- penalizacao de rotacao excessiva;
- penalizacao por obstaculos proximos;
- penalizacao por risco de precipicio;
- recompensa por recuperar da borda;
- penalizacao por avancar em direcao ao cliff;
- recompensa por recuar/virar em situacao de cliff;
- penalizacoes terminais por colisao, queda e cliff/stuck.

A area coberta e calculada atraves de uma grelha de celulas de 0.20 m. Cada
celula nova visitada conta para a area final.


ThymioObstacles - thymio_obstacles.py
Cria obstaculos diretamente no mundo Webots. Em cada reset:

- remove obstaculos anteriores;
- cria 4 obstaculos;
- alterna obstaculos entre zona superior e inferior;
- sorteia posicao, tamanho e orientacao;
- evita que fiquem demasiado proximos entre si.

Isto permite treinar em ambiente random sem editar manualmente o ficheiro .wbt.


TrainingRewardCallback - results.py
Callback usado durante treino. Guarda as recompensas episodicas e calcula:

- media dos primeiros episodios;
- media dos ultimos episodios;
- diferenca entre fim e inicio do treino.

Estas metricas aparecem no results.csv como:

- mean_reward_start;
- mean_reward_end;
- reward_improvement.


SimpleResultsCSV - results.py
Guarda uma linha por experiencia no ficheiro:

results/results.csv

Campos principais:

- experiment;
- model;
- activation;
- environment;
- ablation;
- training_timesteps;
- evaluation_episodes;
- mean_covered_area_m2;
- total_collisions;
- total_falls;
- total_cliff_terminations.


3. MODELO FINAL ESCOLHIDO
-------------------------

O melhor modelo global foi:

ppo_tanh_random_simple_antispin_stuck_50k

Configuracao:

- algoritmo: PPO;
- politica: MlpPolicy;
- ativacao: Tanh;
- ambiente de treino: random;
- timesteps: 50000;
- seeds de avaliacao: 100, 101, 102, 103, 104.

Metricas:

- area media coberta: 0.144 m2;
- colisoes: 0;
- quedas: 0;
- terminacoes por cliff/stuck: 0;
- melhoria de reward: +12.286626.

Observacao visual:
O comportamento foi seguro, mas conservador. O robo tendeu a rodar mais no
centro do ambiente e nao explorou de forma perfeita. Mesmo assim, foi o melhor
compromisso entre area, seguranca e estabilidade.


4. COMPARACOES PRINCIPAIS
-------------------------

PPO Tanh Random
Foi o melhor resultado global. Teve maior area media e zero eventos de seguranca.
E o modelo recomendado como final.

PPO ReLU Random
Metricas piores:

- area media: 0.056 m2;
- colisoes: 1;
- quedas: 0;
- cliff/stuck: 0.

Observacao visual:
Demonstrou um comportamento interessante de recuar quando chegava perto do
precipicio de frente, mas explorou pouco e nao foi tao estavel como Tanh.

RecurrentPPO Tanh Random
Metricas:

- area media: 0.072 m2;
- colisoes: 1;
- quedas: 3;
- cliff/stuck: 1.

Observacao visual:
Tendia a andar em frente com uma curva para a direita e ainda caia. Apesar de a
memoria LSTM poder ser util teoricamente, com o tempo de treino usado nao foi
melhor que PPO simples.

PPO Tanh Standard avaliado em Random
Metricas:

- area media: 0.096 m2;
- colisoes: 0;
- quedas: 0;
- cliff/stuck: 5.

Observacao visual:
Andava muitas vezes de costas e terminou varias vezes por stuck/cliff. Isto
sugere que treinar sempre no mesmo layout standard generalizou pior para os
cenarios random.

PPO Tanh v3.1
Versao mais simples com cliff warning mais agressivo.

Metricas:

- area media: 0.096 m2;
- colisoes: 0;
- quedas: 0;
- cliff/stuck: 3.

Observacao visual:
Fazia curvas, mas terminava cedo por cliff warning. Serve como baseline de
seguranca rigida.


5. ESTUDOS DE ABLACAO
---------------------

Foram feitos estudos de ablacao removendo grupos da reward, mantendo PPO, Tanh,
ambiente random, 50000 timesteps e as mesmas seeds de avaliacao.


Modelo completo
Termos removidos: nenhum.

Metricas:

- area: 0.144 m2;
- colisoes: 0;
- quedas: 0;
- cliff/stuck: 0.

Conclusao:
Melhor equilibrio geral, embora conservador visualmente.


No exploration
Termos removidos:

- new_cell_reward = 0;
- revisit_penalty = 0.

Metricas:

- area: 0.072 m2;
- colisoes: 2;
- quedas: 0;
- cliff/stuck: 0.

Observacao visual:
Andou de forma mais natural e reagiu ao precipicio, mas perdeu incentivo para
cobrir area e teve mais colisoes.

Conclusao:
A recompensa de exploracao e importante para cobertura espacial e estabilidade.


No cliff recovery
Termos removidos:

- cliff_warning_penalty = 0;
- ground_recovery_scale = 0;
- dangerous_forward_penalty_scale = 0;
- reverse_recovery_scale = 0;
- turn_away_scale = 0.

Metricas:

- area: 0.144 m2;
- colisoes: 0;
- quedas: 0;
- cliff/stuck: 0.

Observacao visual:
Apesar das metricas boas, o robo rodava sobre uma roda, sem navegacao real.

Conclusao:
As metricas quantitativas sozinhas podem enganar. A observacao visual mostrou
uma politica degenerada.


No motion shaping
Termos removidos:

- max_forward_reward = 0;
- rotation_penalty_scale = 0.

Metricas:

- area: 0.136 m2;
- colisoes: 0;
- quedas: 0;
- cliff/stuck: 0.

Observacao visual:
O robo ficou a rodar sobre o proprio eixo, praticamente no mesmo sitio.

Conclusao:
O incentivo de movimento e a penalizacao de rotacao ajudam a evitar politicas
locais de rotacao.


No obstacle penalty
Termo removido:

- obstacle_penalty_scale = 0.

Metricas:

- area: 0.072 m2;
- colisoes: 0;
- quedas: 0;
- cliff/stuck: 0.

Observacao visual:
O robo ficava perto do precipicio num ciclo de avancar e recuar, sem exploracao
consistente.

Conclusao:
A penalizacao de obstaculos nao serve apenas para reduzir colisoes. Tambem ajuda
a orientar melhor a navegacao.


Conclusao das ablacoees:
A reward completa foi a mais equilibrada. Algumas ablacoees tiveram metricas
aparentemente boas, mas comportamento visual fraco. Por isso, no relatorio e
importante combinar resultados numericos com observacao qualitativa.


6. METRICAS E GRAFICOS
----------------------

Os resultados estao em:

results/results.csv

Os graficos foram gerados no notebook:

results/report_charts_notebook.ipynb

As imagens finais estao em:

results/figures/

Ficheiros mais importantes:

- 01_area_media_modelos.png;
- 02_eventos_seguranca.png;
- 03_area_vs_eventos.png;
- 04_reward_inicio_fim.png;
- 05_ablation_area_media.png;
- 06_ablation_eventos_seguranca.png;
- 07_ablation_area_vs_eventos.png;
- tabela_resumo_modelos.csv;
- tabela_ablation_reward.csv.

Grafico mais importante para comparacoes gerais:

- 03_area_vs_eventos.png.

Graficos mais importantes para ablacoees:

- 05_ablation_area_media.png;
- 06_ablation_eventos_seguranca.png;
- 07_ablation_area_vs_eventos.png.


7. LIMITACOES OBSERVADAS
------------------------

1. O melhor modelo ainda e conservador.
O robo nao aprendeu uma exploracao perfeita. A politica final e segura, mas
tende a rodar no centro.

2. Area coberta nem sempre significa boa exploracao.
Algumas ablacoees reportaram area alta, mas visualmente o robo apenas rodava no
sitio ou sobre uma roda.

3. Detecao de stuck incompleta.
Quando o robo fica com uma roda fora mas ainda tem pequeno movimento residual, a
logica atual pode nao classificar como stuck.

4. RecurrentPPO nao foi melhor neste contexto.
Apesar de poder usar memoria, precisaria provavelmente de mais treino e ajuste.

5. Velocidade conservadora.
Foi usado speed_factor=0.50, o que torna o comportamento mais seguro mas pode
reduzir a exploracao dentro do tempo maximo do episodio.


8. NOTA IMPORTANTE PARA QUEM FOR CORRER O CODIGO
------------------------------------------------

O ficheiro thymio_rl_controller.py pode estar configurado para a ultima
experiencia executada. Antes de correr o modelo final, confirmar:

experiment_name = "ppo_tanh_random_simple_antispin_stuck_50k"
model_type = "PPO"
activation_name = "Tanh"
environment_name = "random"
ablation_name = "none"

E na reward completa confirmar:

new_cell_reward = 1.00
revisit_penalty = -0.02
max_forward_reward = 0.02
obstacle_penalty_scale = 0.30
rotation_penalty_scale = 0.02
cliff_warning_penalty = -0.20
ground_recovery_scale = 3.00
dangerous_forward_penalty_scale = 2.00
reverse_recovery_scale = 0.20
turn_away_scale = 0.20
collision_penalty = -50.00
fall_penalty = -150.00
cliff_penalty = -80.00


9. FRASE RESUMO PARA O RELATORIO
--------------------------------

O modelo final escolhido foi PPO com ativacao Tanh, treinado em ambiente random.
Embora tenha apresentado comportamento conservador e seja bastante simples, foi o melhor compromisso dos  váriosmodelos testados
entre cobertura, ausencia de colisoes, ausencia de quedas e robustez nos
episodios de avaliacao. Os estudos de ablacao mostraram que remover componentes
da reward pode gerar metricas aparentemente boas, mas tambem comportamentos
degenerados, reforcando a importancia de analisar os resultados quantitativos em
conjunto com a observacao visual.

