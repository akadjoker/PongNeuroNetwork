import pygame
import sys
import random
from ann import ANN

 
pygame.init()

# Constantes do jogo
WIDTH, HEIGHT = 800, 600
PADDLE_WIDTH, PADDLE_HEIGHT = 100, 15
BALL_SIZE = 15
PADDLE_SPEED = 10
BALL_SPEED_X = 5
BALL_SPEED_Y = 5
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
 

 
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Neural Network Pong Luis Santos")
clock = pygame.time.Clock()

 
class Paddle:
    def __init__(self):
        self.rect = pygame.Rect(WIDTH // 2 - PADDLE_WIDTH // 2, HEIGHT - 30, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.speed = PADDLE_SPEED
        self.score = 0
        self.misses = 0
        self.games = 1
    
    def move(self, direction):
        """Move a pad na direção especificada"""
        # Direction é um valor entre -1 (esquerda) e 1 (direita)
        # Limitamos a -1, 0, 1 para movimentos mais estáveis
        if abs(direction) < 0.1:
            direction = 0
        elif direction > 0:
            direction = min(direction, 1.0)
        else:
            direction = max(direction, -1.0)
            
        self.rect.x += direction * self.speed
        
 
        self.rect.x = max(0, min(WIDTH - PADDLE_WIDTH, self.rect.x))
    
    def draw(self, is_ai=False):
        color = GREEN if is_ai else WHITE
        pygame.draw.rect(screen, color, self.rect)

 
class Ball:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.rect = pygame.Rect(WIDTH // 2 - BALL_SIZE // 2, 
                              HEIGHT // 2 - BALL_SIZE // 2,
                              BALL_SIZE, BALL_SIZE)
        self.dx = BALL_SPEED_X * random.choice([-1, 1])
        self.dy = BALL_SPEED_Y
        self.hit_paddle = False
        self.missed_paddle = False
    
    def update(self, paddle):
 
        self.hit_paddle = False
        self.missed_paddle = False
        
 
        self.rect.x += self.dx
        self.rect.y += self.dy
        
        # Colisão com as bordas laterais
        if self.rect.left <= 0 or self.rect.right >= WIDTH:
            self.dx *= -1
        
        # Colisão com a borda superior
        if self.rect.top <= 0:
            self.dy = abs(self.dy)  # Vai para baixo
        
        # Colisão com a pad
        if self.rect.colliderect(paddle.rect) and self.dy > 0:
            self.dy = -abs(self.dy)  # Vai para cima
            
            # Ajusta a direção X baseada em onde a bola atingiu a pad
            # Quanto mais longe do centro, maior o ângulo
            relative_intersect_x = (paddle.rect.centerx - self.rect.centerx) / (PADDLE_WIDTH / 2)
            self.dx = -relative_intersect_x * BALL_SPEED_X
            
            self.hit_paddle = True
            paddle.score += 1
        
        # Bola saiu pela parte inferior da tela
        if self.rect.top >= HEIGHT:
            self.missed_paddle = True
            paddle.misses += 1
            paddle.games += 1
            self.reset()
    
    def draw(self):
        pygame.draw.rect(screen, WHITE, self.rect)

    def predict_landing_position(self):
        """Prevê onde a bola vai atingir a linha da pad"""
        # Se a bola está subindo, não vai atingir a linha da pad nesta trajetória
        if self.dy < 0:
            return None
        
        # Calcula quanto tempo levará para a bola atingir a linha da pad
        time_to_paddle = (HEIGHT - 30 - self.rect.centery) / self.dy
        
        # Calcula a posição X quando a bola atingir a linha da pad
        landing_x = self.rect.centerx + self.dx * time_to_paddle
        
        # Considera os rebotes nas paredes laterais
        effective_width = WIDTH - BALL_SIZE
        
        # Ajusta a posição considerando múltiplos rebotes
        while landing_x < 0 or landing_x > effective_width:
            if landing_x < 0:
                landing_x = -landing_x  # Rebate da parede esquerda
            elif landing_x > effective_width:
                landing_x = 2*effective_width - landing_x  # Rebate da parede direita
        
        return landing_x

class AIController:
    def __init__(self):
        # Configura a rede neural  6 entradas(bola_x, bola_y, bola_dx, bola_dy, paddle_x), 1 saída(direção), 1 camada oculta com 4 neurônios
        self.ann = ANN(6, 1, 1, 4, 0.11)
    
    def predict(self, ball, paddle):
        # Se a bola estiver para cima, não faz nada
        if ball.dy < 0:
            return 0.0
        
        # Normaliza os dados para a rede neural 
        bola_x_norm = ball.rect.centerx / WIDTH * 2 - 1
        bola_y_norm = ball.rect.centery / HEIGHT * 2 - 1
        bola_dx_norm = ball.dx / 10
        bola_dy_norm = ball.dy / 10
        paddle_x_norm = paddle.rect.centerx / WIDTH * 2 - 1
        paddle_y_norm = (HEIGHT - 30) / HEIGHT * 2 - 1  # Posição Y fixa do paddle
        
        # Prepara as entradas para a rede neural
        inputs = [
            bola_x_norm,
            bola_y_norm,
            bola_dx_norm,
            bola_dy_norm,
            paddle_x_norm,
            paddle_y_norm
        ]
        
        # Obtém a saída da rede neural
        output = self.ann.calculate_output(inputs)
        
        # A saída é o movimento do paddle
        return output[0]
    
    def load_model(self, filename="paddle_ai_model.txt"):
 
        return self.ann.load_weights(filename)

 
def draw_text(text, size, x, y, color=WHITE):
    font = pygame.font.Font(None, size)
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect(center=(x, y))
    screen.blit(text_surface, text_rect)
 
def main():
 
    paddle = Paddle()
    ball = Ball()
    ai = AIController()
    
 
    try:
        result = ai.load_model("paddle_ai_model.txt")
        print(result)
        model_loaded = True
    except:
        print("ERRO: Modelo não encontrado. ")
        model_loaded = False
    
 
    ai_controls = True
    show_prediction = True
    paused = False
    
 
   
    hits= 0
    misses= 0,
    hit_rate= 0

    
    # Loop principal do jogo
    running = True
    FPS = 60
    while running:
        
        # Processa eventos
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_p:
                    paused = not paused
                elif event.key == pygame.K_v:
                    show_prediction = not show_prediction
                elif event.key == pygame.K_f:
                    if FPS == 1000:
                        FPS = 60
                    else:
                        FPS = 1000

                elif event.key == pygame.K_r:
                    # Reinicia as estatísticas
                    hits = 0
                    misses = 0
                    hit_rate = 0
                    paddle.score = 0
                    paddle.misses = 0
        
        if paused:
            # Mostrar menu de pausa
            screen.fill(BLACK)
            draw_text("JOGO PAUSADO", 50, WIDTH // 2, HEIGHT // 4)
            draw_text("ESC - Sair", 30, WIDTH // 2, HEIGHT // 2)
            pygame.display.flip()
            clock.tick(FPS)
            continue
        
        # Estado da bola antes da atualização
        previous_hit_paddle = ball.hit_paddle
        previous_missed_paddle = ball.missed_paddle
        
        # Controle da pad

        direction = ai.predict(ball, paddle)
        paddle.move(direction)

        
        # Atualiza a posição da bola e verifica colisões
        ball.update(paddle)
        
        
        if paddle.score >= 1:
            hit_rate = (paddle.score / (paddle.score + paddle.misses)) * 100

 
        
        # Renderiza o jogo
        screen.fill(BLACK)
        
        # Desenha a linha de previsão
        if show_prediction:
            landing_x = ball.predict_landing_position()
            if landing_x is not None and ball.dy > 0:
                pygame.draw.line(screen, (100, 100, 255), 
                                (ball.rect.centerx, ball.rect.centery), 
                                (landing_x, HEIGHT - 30), 1)
        
        # Desenha a pad e a bola
        paddle.draw(is_ai=ai_controls)
        ball.draw()
        
        # Desenha a pontuação
        draw_text(f"Pontos: {paddle.score}   Falhas: {paddle.misses}  Tentativas: {paddle.games}", 30, WIDTH // 2, 30)
        
 
        if model_loaded:
            draw_text(f"Estatísticas: {hit_rate :.1f}% ({paddle.score}/{paddle.games})", 24, WIDTH // 2, 50, GREEN)
            
        else:
            draw_text("Modo: IA (ERRO: Modelo não carregado)", 24, WIDTH // 2, 60, RED)

    
 
        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()