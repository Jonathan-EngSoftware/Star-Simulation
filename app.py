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
# Função para as equações diferenciais
def derivadas(t, y):
    raio, massa, luminosidade, temperatura = y

    # Equações diferenciais corrigidas
    derivadas_raio = np.sqrt(luminosidade / (4 * np.pi * gravidade * massa))
    derivadas_massa = -luminosidade / (gravidade * massa / raio)
    derivadas_luminosidade = 4 * np.pi * raio**2 * temperatura**4
    derivadas_temperatura = -3 * luminosidade / (16 * np.pi * gravidade * raio**2)
    
    return [derivadas_raio, derivadas_massa, derivadas_luminosidade, derivadas_temperatura]

# Simulação usando o método de Runge-Kutta de 4ª ordem
for i, tempo in enumerate(tempos):
    y = [raio_estelar * raio_solar, massa_estelar * massa_solar, luminosidade_estelar, temperatura_estelar]
    k1 = np.multiply(passo_temporal, derivadas(tempo, y))
    k2 = np.multiply(passo_temporal, derivadas(tempo + passo_temporal / 2, np.add(y, np.multiply(0.5, k1))))
    k3 = np.multiply(passo_temporal, derivadas(tempo + passo_temporal / 2, np.add(y, np.multiply(0.5, k2))))
    k4 = np.multiply(passo_temporal, derivadas(tempo + passo_temporal, np.add(y, k3)))

    y = np.add(y, np.multiply(1/6, np.add(np.add(k1, np.multiply(2, k2)), np.add(np.multiply(2, k3), k4))))

    raios[i] = y[0] / raio_solar
    massas[i] = y[1] / massa_solar
    luminosidades[i] = y[2]
    temperaturas[i] = y[3]
    taxas_fusao[i] = 2.5e18 * (y[3] / 1e6)**4 * np.exp(-1.34e8 / (y[3]))  # Correção aqui

# Configuração da animação
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Configuração dos subplots
axs[0, 0].set_xlim(0, tempo_simulacao)
axs[0, 0].set_ylim(0, max(raios) * 1.1)
axs[0, 0].set_xlabel('Tempo (anos)')
axs[0, 0].set_ylabel('Raio Estelar (Raios Solares)')

axs[0, 1].set_xlim(0, tempo_simulacao)
axs[0, 1].set_ylim(0, max(massas) * 1.1)
axs[0, 1].set_xlabel('Tempo (anos)')
axs[0, 1].set_ylabel('Massa Estelar (Massas Solares)')

axs[1, 0].set_xlim(0, tempo_simulacao)
axs[1, 0].set_ylim(0, max(luminosidades) * 1.1)
axs[1, 0].set_xlabel('Tempo (anos)')
axs[1, 0].set_ylabel('Luminosidade Estelar (Watts)')

axs[1, 1].set_xlim(0, tempo_simulacao)
axs[1, 1].set_ylim(0, max(temperaturas) * 1.1)
axs[1, 1].set_xlabel('Tempo (anos)')
axs[1, 1].set_ylabel('Temperatura Estelar (Kelvin)')

# Linhas iniciais (vazias)
lines = [ax.plot([], [])[0] for ax in axs.flatten()]

# Animação
def init():
    for line in lines:
        line.set_data([], [])
    return lines

def update(frame):
    x_data = tempos[:frame]

    y_data = [raios[:frame], massas[:frame], luminosidades[:frame], temperaturas[:frame]]

    for line, y in zip(lines, y_data):
        line.set_data(x_data, y)

    return lines

animation = FuncAnimation(fig, update, frames=len(tempos), init_func=init, blit=True)

plt.suptitle('Animação da Evolução Estelar', y=1.02)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
