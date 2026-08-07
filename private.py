import streamlit as st
import math

# Definición de variables de prueba para verificar los cálculos
# Proyectil .308 Winchester estándar
v0 = 800.0       # m/s (Velocidad inicial)
m_g = 11.34      # gramos (175 grains)
m = m_g / 1000.0 # kg
g = 9.80665      # m/s^2
R = 500.0        # metros (Distancia)
BC = 0.505       # Coeficiente balístico (G1)
rho = 1.225      # kg/m^3 (Densidad del aire estándar)

# Área transversal aproximada para calibre 7.62mm (.308)
d = 0.00782 # metros
A = math.pi * (d/2)**2
# Factor de forma i basado en el peso, BC y dimensiones no es directo, 
# pero podemos aproximar una fuerza de arrastre simplificada o usar el modelo de BC.
# El retraso por arrastre se puede aproximar para la fórmula física.

# Tiempo de vuelo en el vacío vs con resistencia
# Usando aproximación de Pejsa simplificada para el tiempo de vuelo:
# t = R / (v0 * (1 - (R / (2 * D_p)))) donde D_p está ligado al BC.
# Para propósitos de una fórmula física clara en la respuesta, presentaremos las ecuaciones de movimiento con arrastre.

print(f"Masa: {m} kg, Distancia: {R} m, V0: {v0} m/s")
