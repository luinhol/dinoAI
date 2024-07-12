import pygame
import os
import random
import time
from sys import exit
from scipy import stats
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import normalize

pygame.init()

# Valid values: HUMAN_MODE or AI_MODE
GAME_MODE = "AI_MODE"
RENDER_GAME = False

# Global Constants
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 1100
if RENDER_GAME:
    SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

RUNNING = [pygame.image.load(os.path.join("Assets/Dino", "DinoRun1.png")),
           pygame.image.load(os.path.join("Assets/Dino", "DinoRun2.png"))]
JUMPING = pygame.image.load(os.path.join("Assets/Dino", "DinoJump.png"))
DUCKING = [pygame.image.load(os.path.join("Assets/Dino", "DinoDuck1.png")),
           pygame.image.load(os.path.join("Assets/Dino", "DinoDuck2.png"))]

SMALL_CACTUS = [pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus1.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus2.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus3.png"))]
LARGE_CACTUS = [pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus1.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus2.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus3.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus4.png"))]

BIRD = [pygame.image.load(os.path.join("Assets/Bird", "Bird1.png")),
        pygame.image.load(os.path.join("Assets/Bird", "Bird2.png"))]

CLOUD = pygame.image.load(os.path.join("Assets/Other", "Cloud.png"))

BG = pygame.image.load(os.path.join("Assets/Other", "Track.png"))


class Dinosaur:
    X_POS = 90
    Y_POS = 330
    Y_POS_DUCK = 355
    JUMP_VEL = 17
    JUMP_GRAV = 1.1

    def __init__(self):
        self.duck_img = DUCKING
        self.run_img = RUNNING
        self.jump_img = JUMPING

        self.dino_duck = False
        self.dino_run = True
        self.dino_jump = False

        self.step_index = 0
        self.jump_vel = 0
        self.jump_grav = self.JUMP_VEL
        self.image = self.run_img[0]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    def update(self, userInput):
        if self.dino_duck and not self.dino_jump:
            self.duck()
        if self.dino_run:
            self.run()
        if self.dino_jump:
            self.jump()

        if self.step_index >= 20:
            self.step_index = 0

        if userInput == "K_UP" and not self.dino_jump:
            self.dino_duck = False
            self.dino_run = False
            self.dino_jump = True
        elif userInput == "K_DOWN" and not self.dino_jump:
            self.dino_duck = True
            self.dino_run = False
            self.dino_jump = False
        elif userInput == "K_DOWN":
            self.dino_duck = True
            self.dino_run = False
            self.dino_jump = True
        elif not (self.dino_jump or userInput == "K_DOWN"):
            self.dino_duck = False
            self.dino_run = True
            self.dino_jump = False

    def duck(self):
        self.image = self.duck_img[self.step_index // 10]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS_DUCK
        self.step_index += 1

    def run(self):
        self.image = self.run_img[self.step_index // 10]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS
        self.step_index += 1

    def jump(self):
        self.image = self.jump_img
        if self.dino_duck:
            self.jump_grav = self.JUMP_GRAV * 4
        if self.dino_jump:
            self.dino_rect.y -= self.jump_vel
            self.jump_vel -= self.jump_grav
        if self.dino_rect.y > self.Y_POS + 10:
            self.dino_jump = False
            self.jump_vel = self.JUMP_VEL
            self.jump_grav = self.JUMP_GRAV
            self.dino_rect.y = self.Y_POS

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.dino_rect.x, self.dino_rect.y))
        pygame.draw.rect(SCREEN, self.color,
                         (self.dino_rect.x, self.dino_rect.y, self.dino_rect.width, self.dino_rect.height), 2)


    def getXY(self):
        return (self.dino_rect.x, self.dino_rect.y)


class Cloud:
    def __init__(self):
        self.x = SCREEN_WIDTH + random.randint(800, 1000)
        self.y = random.randint(50, 100)
        self.image = CLOUD
        self.width = self.image.get_width()

    def update(self):
        self.x -= game_speed
        if self.x < -self.width:
            self.x = SCREEN_WIDTH + random.randint(2500, 3000)
            self.y = random.randint(50, 100)

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.x, self.y))


class Obstacle():
    def __init__(self, image, type):
        super().__init__()
        self.image = image
        self.type = type
        self.rect = self.image[self.type].get_rect()

        self.rect.x = SCREEN_WIDTH

    def update(self):
        self.rect.x -= game_speed
        if self.rect.x < - self.rect.width:
            obstacles.pop(0)

    def draw(self, SCREEN):
        SCREEN.blit(self.image[self.type], self.rect)

    def getXY(self):
        return (self.rect.x, self.rect.y)

    def getHeight(self):
        return y_pos_bg - self.rect.y

    def getType(self):
        return (self.type)


class SmallCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        self.rect.y = 345


class LargeCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        self.rect.y = 325


class Bird(Obstacle):
    def __init__(self, image):
        self.type = 0
        super().__init__(image, self.type)

        # High, middle or ground
        if random.randint(0, 3) == 0:
            self.rect.y = 345
        elif random.randint(0, 2) == 0:
            self.rect.y = 260
        else:
            self.rect.y = 300
        self.index = 0

    def draw(self, SCREEN):
        if self.index >= 19:
            self.index = 0
        SCREEN.blit(self.image[self.index // 10], self.rect)
        self.index += 1


class KeyClassifier:
    def __init__(self, state):
        pass

    def keySelector(self, distance, obHeight, speed, obType, nextObDistance, nextObHeight, nextObType):
        pass

    def updateState(self, state):
        pass


def first(x):
    return x[0]

obstacle_name = {"LargeCactus": 0, "SmallCactus": 1, "Bird": 2}

class MyKeyClassifier(KeyClassifier):
    def __init__(self, state):
        self.matriz_pesos_1_camada = state[0:28].reshape(4,7)
        self.matriz_pesos_2_camada = state[28:32].reshape(1,4)
        # self.matriz_pesos_1_camada = state[0:16].reshape(4,4)
        # self.matriz_pesos_2_camada = state[16:20].reshape(1,4)

    def keySelector(self, distance, obHeight, speed, obType, nextObDistance, nextObHeight, nextObType):
        
        if (isinstance(obType, int)):
            obType = -1
        else:
            obType = obstacle_name[type(obType).__name__]
            
        if (isinstance(nextObType, int)):
            nextObType = -1
        else:
            nextObType = obstacle_name[type(nextObType).__name__]
            
        saida_neuronios_entrada = np.array([distance, obHeight, speed, obType, nextObDistance, nextObHeight, nextObType])
        # saida_neuronios_entrada = np.array([distance, obHeight, speed, obType])
        
        coluna_entradas = saida_neuronios_entrada.reshape(7,1)
        # coluna_entradas = saida_neuronios_entrada.reshape(4,1)

        # saida linear dos neuronios intermediarios e o produto entre a matriz de pesos da 1 camada e o vetor coluna de entrada
        saida_linear_neuronios_intermediarios = np.dot(self.matriz_pesos_1_camada, coluna_entradas)
        # saida de ativacao dos neuronios intermediarios e a funcao relu aplicada sobre o vetor de saida linear
        saida_ativacao_neuronios_intermediarios = np.maximum(0, saida_linear_neuronios_intermediarios)

        # saida linear do neuronio de saida e o produto entre a matriz de pesos da 2 camada e o vetor com a saida dos neuronios intermediarios
        saida_linear_neuronio_saida = np.dot(self.matriz_pesos_2_camada, saida_ativacao_neuronios_intermediarios)
        # saida de ativacao do neuronio de saida e a funcao sigmoid aplicada a saida linear
        saida_ativacao_neuronio_saida = expit(saida_linear_neuronio_saida)
                                
        # se a saida linear do neuronio de saida for superior a 0.55, o dinossauro pula
        if (saida_ativacao_neuronio_saida >= 0.55):
            return "K_UP"
        else:
            return "K_DOWN"

    def updateState(self, state):
        self.weights = state

def convergent(population):
    conv = False
    if len(population) > 0:
        base = population[0]
        i = 0
        while i < len(population):
            if not np.array_equal(base, population[i]):
                return False
            i += 1
        return True

class Swarm():
    def __init__(self, n_particles, n_weights, melhor_pontuacao_enxame, melhor_posicao_enxame):
        self.n_particles = n_particles
        
        # gera um vetor posicao e um velocidade com valores aleatorios entre -1 e 1
        # self.posicao = np.random.rand(n_particles, n_weights) * 2 - 1
        # self.velocidade = np.random.rand(n_particles, n_weights) * 2 - 1
        
        
        self.posicao = np.random.rand(n_particles, n_weights) * 2 - 1
        self.velocidade = np.random.rand(n_particles, n_weights) * 2 - 1
        
        # declara variaveis com as melhores posicoes e pontuacoes das particulas, alem da melhor pontuacao do enxame
        self.melhor_posicao_particula = self.posicao.copy()
        self.melhor_pontuacao_particula = [0] * n_particles
        self.melhor_pontuacao_enxame = melhor_pontuacao_enxame
        self.melhor_posicao_enxame = melhor_posicao_enxame.copy()
                
        # cria um classificador com os pesos (posicao) do individuo 1, e testa o jogo com ele
        weights = self.posicao.copy()
        results  = manyPlaysResultsTrain(10, weights)
        
        # se a pontuacao dessa particula foi a melhor dela ate o momento, salva no vetor de melhor pontuacao (como esta iniciando, com certeza vai ser)
        self.melhor_pontuacao_particula = results.copy()
            
        for i in range(self.n_particles):
            # se a pontuacao dessa particula foi a melhor do enxame ate agora, salva na variavel do enxame
            if self.melhor_pontuacao_particula[i] > self.melhor_pontuacao_enxame:
                self.melhor_posicao_enxame = self.posicao[i].copy()
                self.melhor_pontuacao_enxame = self.melhor_pontuacao_particula[i]
        
        # salva coeficientes usados no calculo de velocidade
        self.w = 1
        self.c1 = 1.2
        self.c2 = 1.2
                
    def update_swarm(self):
        # print("posicao", self.posicao)
        # cria dois vetores do tamanho do numero de particulas com valores aleatorios entre 0 e 1
        r1 = np.random.rand(self.n_particles)
        r2 = np.random.rand(self.n_particles)
        
        # atualiza vetor de velocidades
        self.velocidade = np.dot(self.w,self.velocidade) + ((self.melhor_posicao_particula - self.posicao) * np.dot(self.c1, r1)[:, np.newaxis]) + ((self.melhor_posicao_enxame - self.posicao) * np.dot(self.c2, r2)[:, np.newaxis])
        # self.velocidade = (self.w * self.velocidade) + np.dot((self.c1 * r1) , (self.melhor_posicao_particula - self.posicao)) + np.dot((self.c2 * r2) , (self.melhor_posicao_enxame - self.posicao))
        
        self.velocidade = np.clip(self.velocidade, -10, 10)
        
        # atualiza vetor posicoes
        self.posicao = self.posicao + self.velocidade
        # print("posicao", self.posicao)
        
        # roda n classificadores com as posicoes
        # results = []
        # weights = normalize(self.posicao, norm="max")
        weights = self.posicao.copy()
        results = manyPlaysResultsTrain(10, weights)
        # print("results", results)
        
        for i in range(self.n_particles):
            # cria um classificador com os pesos (posicao) do individuo 1, e testa o jogo com ele
            # aiPlayer = MyKeyClassifier(self.posicao[i])
            
            # se a pontuacao dessa particula foi a melhor dela ate o momento, salva no vetor de melhor pontuacao
            if results[i] > self.melhor_pontuacao_particula[i]:
                self.melhor_posicao_particula[i] = self.posicao[i].copy()
                self.melhor_pontuacao_particula[i] = results[i]
            
            # se a pontuacao dessa particula foi a melhor do enxame ate agora, salva na variavel do enxame
            if results[i] > self.melhor_pontuacao_enxame:
                self.melhor_posicao_enxame = self.posicao[i].copy()
                self.melhor_pontuacao_enxame = results[i]
                
        
    def get_population(self):
        return self.posicao.copy()
    
    def get_global_optima(self):
        return self.melhor_posicao_enxame.copy(), self.melhor_pontuacao_enxame

def pso(n_particles, n_weights, max_iter, max_time, melhor_pontuacao_enxame, melhor_posicao_enxame):
    start = time.process_time()
    itera = 0    
    end = 0
    
    # gera enxame
    enxame = Swarm(n_particles, n_weights, melhor_pontuacao_enxame, melhor_posicao_enxame)
    
    conv = convergent(enxame.get_population())

    # enquanto nao convergiu, atingiu tempo ou iteracao maxima
    while not conv and itera < max_iter/10 and end-start <= max_time/200:
        enxame.update_swarm()
        
        # verifica se o valor convergiu
        conv = convergent(enxame.get_population())
        
        itera+=1
        end = time.process_time()
        
        melhor_posicao_enxame, melhor_pontuacao_enxame = enxame.get_global_optima()
        print("iteracao:", itera, " melhor pontuacao:", melhor_pontuacao_enxame, " tempo:", end-start)
    
    melhor_posicao_enxame, melhor_pontuacao_enxame = enxame.get_global_optima()
    return melhor_posicao_enxame.copy(), melhor_pontuacao_enxame, itera

def learn(n_particles, n_weights, max_iter, max_time):
    start = time.process_time()
    itera = 0    
    end = 0
    
    # melhor_posicao_geral = np.array([-5.5747902, -4.34080708, -1.07135268, -5.65339635, -1.14022635,
    #                         1.35129083, 22.3786579, -5.1022398, -4.62575887, -1.92252924,
    #                         -6.25952651, -21.37749674, -4.59973041, 0.73932804, -0.71396757,
    #                         -22.83032348, 2.39480565, 10.34819655, -0.76261333, 1.28879508])
    
    # melhor_pontuacao_geral = 1240.4134557828877
    
    melhor_posicao_geral = []
    melhor_pontuacao_geral = 0

    # enquanto nao atingiu tempo ou iteracao maxima
    while itera < max_iter and end-start <= max_time:
        print(" populacao", itera+1)
        melhor_posicao_enxame, melhor_pontuacao_enxame, iter = pso(n_particles, n_weights, max_iter, max_time, melhor_pontuacao_geral, melhor_posicao_geral)
        
        if (melhor_pontuacao_enxame > melhor_pontuacao_geral):
            melhor_pontuacao_geral = melhor_pontuacao_enxame
            melhor_posicao_geral = melhor_posicao_enxame.copy()
        
        print(melhor_pontuacao_geral, melhor_posicao_geral, melhor_pontuacao_enxame)
        # print(melhor_pontuacao_geral, np.array2string(melhor_posicao_geral, separator=', ', max_line_width = 999999999), melhor_pontuacao_enxame)
        itera+=1
        end = time.process_time()
    
    return melhor_posicao_geral, melhor_pontuacao_geral, itera


def playerKeySelector():
    userInputArray = pygame.key.get_pressed()

    if userInputArray[pygame.K_UP]:
        return "K_UP"
    elif userInputArray[pygame.K_DOWN]:
        return "K_DOWN"
    else:
        return "K_NO"


def playGame(solutions):
    global game_speed, x_pos_bg, y_pos_bg, points, obstacles
    run = True

    clock = pygame.time.Clock()
    cloud = Cloud()
    font = pygame.font.Font('freesansbold.ttf', 20)

    players = []
    players_classifier = []
    solution_fitness = []
    died = []

    game_speed = 10
    x_pos_bg = 0
    y_pos_bg = 383
    points = 0

    obstacles = []
    death_count = 0
    spawn_dist = 0

    for solution in solutions:
        players.append(Dinosaur())
        players_classifier.append(MyKeyClassifier(solution))
        solution_fitness.append(0)
        died.append(False)

    def score():
        global points, game_speed
        points += 0.25
        if points % 100 == 0:
            game_speed += 1

        if RENDER_GAME:
            text = font.render("Points: " + str(int(points)), True, (0, 0, 0))
            textRect = text.get_rect()
            textRect.center = (1000, 40)
            SCREEN.blit(text, textRect)


    def background():
        global x_pos_bg, y_pos_bg
        image_width = BG.get_width()
        SCREEN.blit(BG, (x_pos_bg, y_pos_bg))
        SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
        if x_pos_bg <= -image_width:
            SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
            x_pos_bg = 0
        x_pos_bg -= game_speed

    def statistics():
        text_1 = font.render(f'Dinosaurs Alive:  {str(died.count(False))}', True, (0, 0, 0))
        text_3 = font.render(f'Game Speed:  {str(game_speed)}', True, (0, 0, 0))

        SCREEN.blit(text_1, (50, 450))
        SCREEN.blit(text_3, (50, 480))

    while run and (False in died):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                exit()

        if RENDER_GAME:
            SCREEN.fill((255, 255, 255))

        for i,player in enumerate(players):
            if not died[i]:
                distance = 1500
                nextObDistance = 2000
                obHeight = 0
                nextObHeight = 0
                obType = 2
                nextObType = 2
                if len(obstacles) != 0:
                    xy = obstacles[0].getXY()
                    distance = xy[0]
                    obHeight = obstacles[0].getHeight()
                    obType = obstacles[0]

                if len(obstacles) == 2:
                    nextxy = obstacles[1].getXY()
                    nextObDistance = nextxy[0]
                    nextObHeight = obstacles[1].getHeight()
                    nextObType = obstacles[1]

                userInput = players_classifier[i].keySelector(distance, obHeight, game_speed, obType, nextObDistance, nextObHeight,nextObType)

                player.update(userInput)

                if RENDER_GAME:
                    player.draw(SCREEN)

        if len(obstacles) == 0 or obstacles[-1].getXY()[0] < spawn_dist:
            spawn_dist = random.randint(0, 670)
            if random.randint(0, 2) == 0:
                obstacles.append(SmallCactus(SMALL_CACTUS))
            elif random.randint(0, 2) == 1:
                obstacles.append(LargeCactus(LARGE_CACTUS))
            elif random.randint(0, 5) == 5:
                obstacles.append(Bird(BIRD))

        for obstacle in list(obstacles):
            obstacle.update()
            if RENDER_GAME:
                obstacle.draw(SCREEN)


        if RENDER_GAME:
            background()
            statistics()
            cloud.draw(SCREEN)

        cloud.update()

        score()

        if RENDER_GAME:
            clock.tick(60)
            pygame.display.update()

        for obstacle in obstacles:
            for i, player in enumerate(players):
                if player.dino_rect.colliderect(obstacle.rect) and died[i] == False:
                    solution_fitness[i] = points
                    died[i] = True

    return solution_fitness

def manyPlaysResultsTrain(rounds,solutions):
    results = []

    for round in range(rounds):
        results += [playGame(solutions)]

    npResults = np.asarray(results)

    mean_results = np.mean(npResults,axis = 0) - np.std(npResults,axis=0) # axis 0 calcula media da coluna
    return mean_results


def manyPlaysResultsTest(rounds,best_solution):
    results = []
    for round in range(rounds):
        results += [playGame([best_solution])[0]]

    npResults = np.asarray(results)
    return (results, npResults.mean() - npResults.std())


def main():
    global aiPlayer
    
    n_weights = 20
    n_weights = 32
    n_particles = 100
    max_iter = 1000
    max_time = 43200 # 12 horas
    
    best_weights, best_value, itera = learn(n_particles, n_weights, max_iter, max_time)
    
    # best_weights = normalize(best_weights, norm="max")
    res, value = manyPlaysResultsTest(30, best_weights)
    
    npRes = np.asarray(res)
    print(res, npRes.mean(), npRes.std(), value)


main()