import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constantes
massa_solar = 1.989e30  # em kg
raio_solar = 6.957e8  # em metros
gravidade = 6.67430e-11  # Constante gravitacional em m^3/kg/s^2
luminosidade_solar = 3.828e26  # em Watts
tempo_simulacao = 1e7  # em anos
passo_temporal = 1e4  # em anos

# Condições iniciais
massa_estelar = 10.0  # massa da estrela em massas solares
raio_estelar = 1.0  # raio da estrela em raios solares
luminosidade_estelar = luminosidade_solar
temperatura_estelar = 5778  # em Kelvin

# Arrays para armazenar dados
tempos = np.arange(0, tempo_simulacao, passo_temporal)
raios = np.zeros_like(tempos)
massas = np.zeros_like(tempos)
luminosidades = np.zeros_like(tempos)
temperaturas = np.zeros_like(tempos)
taxas_fusao = np.zeros_like(tempos)

# Função para as equações diferenciais
def derivadas(t, y):
    raio, massa, luminosidade, temperatura = y

    # Equações diferenciais simplificadas
    derivadas_raio = np.sqrt(luminosidade / (4 * np.pi * gravidade * massa))
    derivadas_massa = -luminosidade / (gravidade * massa / raio)
    derivadas_luminosidade = 4 * np.pi * raio**2 * temperatura**4
    derivadas_temperatura = -3 * luminosidade / (16 * np.pi * gravidade * raio**2)
    
    # Correção na equação para a taxa de fusão nuclear
    taxa_fusao = 2.5e18 * (temperatura / 1e6)**4 * np.exp(-1.34e8 / (temperatura))

    return [derivadas_raio, derivadas_massa, derivadas_luminosidade, derivadas_temperatura, taxa_fusao]

# Simulação usando o método de Runge-Kutta de 4ª ordem
for i, tempo in enumerate(tempos):
    y = [raio_estelar * raio_solar, massa_estelar * massa_solar, luminosidade_estelar, temperatura_estelar]
    k1 = np.multiply(passo_temporal, derivadas(tempo, y))
    k2 = np.multiply(passo_temporal, derivadas(tempo + passo_temporal / 2, np.add(y, np.multiply(0.5, k1))))
    k3 = np.multiply(passo_temporal, derivadas(tempo + passo_temporal / 2, np.add(y, np.multiply(0.5, k2))))
    k4 = np.multiply(passo_temporal, derivadas(tempo + passo_temporal, np.add(y, k3)))

    y = np.add(y, np.multiply(1/6, np.add(k1, np.add(2*k2, np.add(2*k3, k4)))))

    raios[i] = y[0] / raio_solar
    massas[i] = y[1] / massa_solar
    luminosidades[i] = y[2]
    temperaturas[i] = y[3]
    taxas_fusao[i] = y[4]

# Configuração da animação
fig, ax = plt.subplots()

ax.set_xlim(0, tempo_simulacao)
ax.set_ylim(0, max(raios) * 1.1)
ax.set_xlabel('Tempo (anos)')
ax.set_ylabel('Raio Estelar (Raios Solares)')

# Linha inicial (vazia)
line, = ax.plot([], [], lw=2)

# Animação
def init():
    line.set_data([], [])
    return line,

def update(frame):
    x_data = tempos[:frame]
    y_data = raios[:frame]

    line.set_data(x_data, y_data)
    return line,

animation = FuncAnimation(fig, update, frames=len(tempos), init_func=init, blit=True)

plt.title('Animação da Evolução Estelar')
plt.show()
