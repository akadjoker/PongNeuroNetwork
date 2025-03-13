import random
import time
import matplotlib.pyplot as plt
from ann import ANN

 
WIDTH = 800
HEIGHT = 600
PADDLE_Y = 550  # Altura do paddle na tela
PADDLE_WIDTH = 100

# Configuração da rede neural 
# 6 entradas, 1 saída, 1 camada oculta com 4 neurônios, taxa de 0.11
rede = ANN(6, 1, 1, 4, 0.1)

def treino(segundos=60):
 
    print(f"Iniciando treinamento   por {segundos} segundos...")
    
    # Marca o tempo inicial
    inicio = time.time()
    amostras = 0
    
    # Variáveis para monitorar o erro
    erro_total = 0
    erros_medios = []
    tempo_pontos = []
    intervalo_erro = 1000
    
    # Posição inicial do paddle (centro da tela)
    paddle_x = WIDTH / 2
    paddle_y = PADDLE_Y
    
    # Loop de treinamento
    while time.time() - inicio < segundos:
        # Gera uma nova posição e velocidade para a bola
        bola_x = random.randint(50, WIDTH-50)
        bola_y = random.randint(50, 300)
        bola_dx = random.uniform(-10, 10)
        bola_dy = random.uniform(3, 10)  # Valor positivo para ir para baixo
        
        # Calcula quando a bola vai atingir a linha do paddle
        if bola_dy > 0:  # Se a bola estiver descendo
            # Tempo até atingir a linha do paddle
            tempo_ate_paddle = (PADDLE_Y - bola_y) / bola_dy
            
            # Ponto de impacto na linha do paddle
            ponto_impacto_x = bola_x + (bola_dx * tempo_ate_paddle)
            
            # Considera rebotes nas paredes laterais
            while ponto_impacto_x < 0 or ponto_impacto_x > WIDTH:
                if ponto_impacto_x < 0:
                    ponto_impacto_x = -ponto_impacto_x
                elif ponto_impacto_x > WIDTH:
                    ponto_impacto_x = 2*WIDTH - ponto_impacto_x
            
            # Diferença vertical entre o ponto de impacto e o paddle
 
            dy = ponto_impacto_x - paddle_x
            
            # Normalização das entradas  
            # Normaliza para valores entre -1 e 1 aproximadamente
            bola_x_norm = bola_x / WIDTH * 2 - 1
            bola_y_norm = bola_y / HEIGHT * 2 - 1
            bola_dx_norm = bola_dx / 10  # Normalizar para ~[-1, 1]
            bola_dy_norm = bola_dy / 10  # Normalizar para ~[0, 1]
            paddle_x_norm = paddle_x / WIDTH * 2 - 1
            paddle_y_norm = paddle_y / HEIGHT * 2 - 1
            
            # Normaliza o alvo (dy)
 
            dy_norm = dy / (WIDTH/2)  # Normaliza para ~[-1, 1]
            
            # Prepara as entradas e saída para a rede neural
            entradas = [
                bola_x_norm,    # posição x da bola normalizada
                bola_y_norm,    # posição y da bola normalizada
                bola_dx_norm,   # velocidade x da bola normalizada
                bola_dy_norm,   # velocidade y da bola normalizada
                paddle_x_norm,  # posição x do paddle normalizada
                paddle_y_norm   # posição y do paddle normalizada
            ]
            
            # Treina a rede
            saida = rede.train(entradas, [dy_norm])
            amostras += 1
            
            # Calcula o erro
            erro = abs(saida[0] - dy_norm)
            erro_total += erro
            
            # Atualiza a posição do paddle baseada na saída da rede
            # (simulando o que aconteceria no jogo)
            paddle_x = ponto_impacto_x
            
            # Mostra progresso a cada intervalo de amostras
            if amostras % intervalo_erro == 0:
                tempo_passado = time.time() - inicio
                erro_medio = erro_total / intervalo_erro
                
                # Armazena para o gráfico
                erros_medios.append(erro_medio)
                tempo_pontos.append(tempo_passado)
                
                print(f"Amostra {amostras}: Erro médio = {erro_medio:.4f} " +
                      f"({amostras/tempo_passado:.1f} amostras/s)")
                
                # Reseta o erro total para o próximo intervalo
                erro_total = 0
    
    # Estatísticas finais
    tempo_total = time.time() - inicio
    print(f"\nTreinamento concluído!")
    print(f"Treinadas {amostras} amostras em {tempo_total:.1f} segundos")
    print(f"Velocidade média: {amostras/tempo_total:.1f} amostras/segundo")
    
    # Salva o modelo treinado
    rede.save_weights("paddle_ai_model.txt")
    print("Modelo salvo como 'paddle_ai_model.txt'")
    
    # Plota o gráfico de erro
    plt.figure(figsize=(10, 6))
    plt.plot(tempo_pontos, erros_medios, 'b-', marker='o')
    plt.title('Evolução do Erro Durante o Treinamento')
    plt.xlabel('Tempo (segundos)')
    plt.ylabel('Erro Médio')
    plt.grid(True)
    plt.savefig('erro_treinamento.png')
    print("Gráfico de erro salvo como 'erro_treinamento.png'")
    
    return erros_medios, tempo_pontos

def teste_modelo(amostras=1000):
    """Testa o modelo treinado com amostras aleatórias"""
    print(f"\nTestando modelo com {amostras} amostras...")
    
    erro_total = 0
    erro_maximo = 0
    acertos = 0
    
    # Posição do paddle
    paddle_x = WIDTH / 2
    paddle_y = PADDLE_Y
    
    for i in range(amostras):
        # Gera uma amostra aleatória
        bola_x = random.randint(50, WIDTH-50)
        bola_y = random.randint(50, 300)
        bola_dx = random.uniform(-10, 10)
        bola_dy = random.uniform(3, 10)
        
        # Calcula quando a bola vai atingir a linha do paddle
        tempo_ate_paddle = (PADDLE_Y - bola_y) / bola_dy
        ponto_impacto_x = bola_x + (bola_dx * tempo_ate_paddle)
        
        # Considera rebotes nas paredes laterais
        while ponto_impacto_x < 0 or ponto_impacto_x > WIDTH:
            if ponto_impacto_x < 0:
                ponto_impacto_x = -ponto_impacto_x
            elif ponto_impacto_x > WIDTH:
                ponto_impacto_x = 2*WIDTH - ponto_impacto_x
        
        # Diferença entre o ponto de impacto e o paddle
        dy = ponto_impacto_x - paddle_x
        
        # Normaliza as entradas
        bola_x_norm = bola_x / WIDTH * 2 - 1
        bola_y_norm = bola_y / HEIGHT * 2 - 1
        bola_dx_norm = bola_dx / 10
        bola_dy_norm = bola_dy / 10
        paddle_x_norm = paddle_x / WIDTH * 2 - 1
        paddle_y_norm = paddle_y / HEIGHT * 2 - 1
        
        # Normaliza o alvo
        dy_norm = dy / (WIDTH/2)
        
        # Prepara as entradas
        entradas = [
            bola_x_norm,
            bola_y_norm,
            bola_dx_norm,
            bola_dy_norm,
            paddle_x_norm,
            paddle_y_norm
        ]
        
        # Obtém a previsão da rede
        saida = rede.calculate_output(entradas)
        
        # Calcula o erro
        erro = abs(saida[0] - dy_norm)
        erro_total += erro
        erro_maximo = max(erro_maximo, erro)
        
        # Considera acerto se o erro for menor que 0.2
        if erro < 0.2:
            acertos += 1
        
        # Atualiza a posição do paddle para o próximo teste
        paddle_x = ponto_impacto_x
    
    erro_medio = erro_total / amostras
    taxa_acerto = (acertos / amostras) * 100
    
    print(f"Teste concluído!")
    print(f"Erro médio: {erro_medio:.4f}")
    print(f"Erro máximo: {erro_maximo:.4f}")
    print(f"Taxa de acerto (erro < 0.2): {taxa_acerto:.1f}%")
    
    # Converte o erro para posição em pixels para ter uma ideia prática
    erro_pixel_medio = (erro_medio / 2) * WIDTH
    erro_pixel_maximo = (erro_maximo / 2) * WIDTH
    
    print(f"Em pixels, isso representa:")
    print(f"Erro médio: {erro_pixel_medio:.1f} pixels")
    print(f"Erro máximo: {erro_pixel_maximo:.1f} pixels")
    
    return erro_medio, erro_maximo, taxa_acerto

if __name__ == "__main__":
 
    segundos = int(input("Duração do treino em segundos (recomendado: 60): "))
    
    erros_medios, tempo_pontos = treino(segundos)
    
    # Testa o modelo treinado
    erro_medio, erro_maximo, taxa_acerto = teste_modelo(1000)
    
    print("\nTreinamento e teste concluídos com sucesso!")
    