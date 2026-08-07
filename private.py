import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Configuración inicial de la página
st.set_page_config(
    page_title="Simulador de Óptica Geométrica - Láser",
    page_icon="💡",
    layout="wide",
)

st.title("💡 Simulador de Óptica con Ley de Reflexión Real")
st.markdown(
    "Simulación exacta basada en la ley de la reflexión ($\theta_i = \theta_r$)"
    " con vectores normales y cálculo de rebotes vectoriales."
)

# --- PANEL DE CONTROL LATERAL ---
st.sidebar.header("🎛️ Controles del Sistema")

# 1. Dimensiones del Campo
st.sidebar.subheader("Dimensiones del Área")
max_x = st.sidebar.slider("Límite Eje X (Ancho)", 10.0, 50.0, 20.0, 1.0)
max_y = st.sidebar.slider("Límite Eje Y (Largo)", 10.0, 50.0, 20.0, 1.0)

# 2. Configuración del Láser
st.sidebar.subheader("Laser (Origen)")
laser_x = st.sidebar.slider(
    "Posición X del Láser", -max_x / 2, max_x / 2, 0.0, 0.5
)
laser_angle_deg = st.sidebar.slider(
    "Ángulo Inicial del Láser (grados)", 0, 180, 90, 1
)

# 3. Configuración del Objetivo
st.sidebar.subheader("🎯 Objetivo")
target_x = st.sidebar.slider(
    "Posición X del Objetivo", -max_x / 2, max_x / 2, 5.0, 0.5
)
target_y = st.sidebar.slider(
    "Posición Y del Objetivo", 0.0, max_y, max_y * 0.8, 0.5
)
target_radius = st.sidebar.slider("Radio del Objetivo", 0.5, 3.0, 1.0, 0.2)

# 4. Configuración de Espejos
st.sidebar.subheader("🪞 Espejos")
num_mirrors = st.sidebar.slider("Cantidad de Espejos", 1, 5, 2)

mirrors = []
for i in range(num_mirrors):
  st.sidebar.markdown(f"**Espejo {i+1}**")
  mx = st.sidebar.slider(
      f"Pos X #{i+1}", -max_x / 2, max_x / 2, float(i * 4 - 2), 0.5, key=f"mx_{i}"
  )
  my = st.sidebar.slider(
      f"Pos Y #{i+1}", 0.0, max_y, float(i * 4 + 4), 0.5, key=f"my_{i}"
  )
  m_ang = st.sidebar.slider(
      f"Ángulo #{i+1} (grados)", 0, 180, 45, 1, key=f"mang_{i}"
  )
  mirrors.append({"x": mx, "y": my, "angle": m_ang})


# --- MOTOR FÍSICO DE REFLEXIÓN REAL ---
def get_ray_path(
    l_x, l_start_deg, mirrors_list, t_x, t_y, t_rad, b_x, b_y, max_bounces=10
):
  path_x = [l_x]
  path_y = [0.0]

  rad = np.radians(l_start_deg)
  dir_x = np.cos(rad)
  dir_y = np.sin(rad)
  curr_x, curr_y = l_x, 0.0

  hit_target = False

  for _ in range(max_bounces):
    closest_t = float("inf")
    next_x, next_y = curr_x + dir_x * 100, curr_y + dir_y * 100
    hit_mirror_idx = -1
    normal_vector = None

    # 1. Comprobación con fronteras (paredes de la caja)
    if dir_x > 0:
      t_wall = (b_x - curr_x) / dir_x
      if 0 < t_wall < closest_t:
        closest_t = t_wall
        next_x, next_y = b_x, curr_y + dir_x * t_wall
        normal_vector = (-1, 0)
    elif dir_x < 0:
      t_wall = (-b_x - curr_x) / dir_x
      if 0 < t_wall < closest_t:
        closest_t = t_wall
        next_x, next_y = -b_x, curr_y + dir_x * t_wall
        normal_vector = (1, 0)

    if dir_y > 0:
      t_wall = (b_y - curr_y) / dir_y
      if 0 < t_wall < closest_t:
        closest_t = t_wall
        next_x, next_y = curr_x + dir_x * t_wall, b_y
        normal_vector = (0, -1)

    # 2. Comprobación con los espejos (segmentos físicos orientados por ángulo)
    mirror_length = 3.0
    for idx, m in enumerate(mirrors_list):
      m_rad = np.radians(m["angle"])
      # Vector tangente del espejo en el plano XY
      m_dx = np.cos(m_rad) * (mirror_length / 2)
      m_dy = np.sin(m_rad) * (mirror_length / 2)

      x1, y1 = m["x"] - m_dx, m["y"] - m_dy
      x2, y2 = m["x"] + m_dx, m["y"] + m_dy

      # Intersección Rayo-Segmento (Cramer / 2D Line Intersection)
      det = dir_x * (y1 - y2) - dir_y * (x1 - x2)
      if abs(det) > 1e-6:
        t = ((x1 - curr_x) * (y1 - y2) - (y1 - curr_y) * (x1 - x2)) / det
        u = ((x1 - curr_x) * dir_y - (y1 - curr_y) * dir_x) / det

        if 0.001 < t < closest_t and 0 <= u <= 1:
          closest_t = t
          next_x, next_y = curr_x + dir_x * t, curr_y + dir_y * t
          hit_mirror_idx = idx

          # Cálculo físico riguroso de la Normal al Espejo:
          # Si la tangente es (m_dx, m_dy), las normales candidatas son (-m_dy, m_dx) o (m_dy, -m_dx)
          nx, ny = -m_dy, m_dx
          length_n = np.hypot(nx, ny)
          nx, ny = nx / length_n, ny / length_n

          # Orientar la normal en sentido opuesto al rayo incidente (producto escalar negativo)
          if nx * dir_x + ny * dir_y > 0:
            nx, ny = -nx, -ny
          normal_vector = (nx, ny)

    # 3. Verificación de intercepción con el objetivo en este segmento
    v_vec = np.array([next_x - curr_x, next_y - curr_y])
    w_vec = np.array([t_x - curr_x, t_y - curr_y])
    v_len_sq = np.dot(v_vec, v_vec)
    if v_len_sq > 0:
      c1 = np.dot(w_vec, v_vec)
      if c1 <= 0:
        proj_dist = np.hypot(t_x - curr_x, t_y - curr_y)
      elif c1 >= v_len_sq:
        proj_dist = np.hypot(t_x - next_x, t_y - next_y)
      else:
        b_val = c1 / v_len_sq
        pb_x = curr_x + b_val * v_vec[0]
        pb_y = curr_y + b_val * v_vec[1]
        proj_dist = np.hypot(t_x - pb_x, t_y - pb_y)

      if proj_dist <= t_rad:
        hit_target = True
        path_x.append(t_x)
        path_y.append(t_y)
        break

    path_x.append(next_x)
    path_y.append(next_y)

    if hit_target or hit_mirror_idx == -1 or normal_vector is None:
      break

    # 4. Ley de Reflexión Vectorial Exacta: R = D - 2(D · N) * N
    d_vec = np.array([dir_x, dir_y])
    n_vec = np.array(normal_vector)
    r_vec = d_vec - 2 * np.dot(d_vec, n_vec) * n_vec
    dir_x, dir_y = r_vec[0], r_vec[1]
    curr_x, curr_y = next_x, next_y

  return path_x, path_y, hit_target


# Calcular la trayectoria con la física actualizada
path_x, path_y, success = get_ray_path(
    laser_x,
    laser_angle_deg,
    mirrors,
    target_x,
    target_y,
    target_radius,
    max_x / 2,
    max_y,
)

if success:
  st.success("🎯 ¡Impacto exitoso con el objetivo!")
else:
  st.warning(
    "⚠️ El láser no alcanzó el objetivo. Mueve los ángulos de los espejos para"
    " desviar el haz."
  )


# --- RENDERIZADO GRÁFICO CON PLOTLY ---
fig = go.Figure()

# 1. Trayectoria del rayo láser
fig.add_trace(
    go.Scatter(
        x=path_x,
        y=path_y,
        mode="lines+markers",
        name="Rayo Láser",
        line=dict(color="#FF0055", width=3),
        marker=dict(size=5, color="#FF5588"),
    )
)

# 2. Posición del Láser
fig.add_trace(
    go.Scatter(
        x=[laser_x],
        y=[0],
        mode="markers+text",
        name="Emisor Láser",
        marker=dict(color="#00FFFF", size=16, symbol="triangle-up"),
        text=["Láser"],
        textposition="bottom center",
    )
)

# 3. Objetivo
theta = np.linspace(0, 2 * np.pi, 100)
t_circle_x = target_x + target_radius * np.cos(theta)
t_circle_y = target_y + target_radius * np.sin(theta)

fig.add_trace(
    go.Scatter(
        x=t_circle_x,
        y=t_circle_y,
        mode="lines",
        name="Objetivo",
        fill="toself",
        fillcolor="rgba(0, 255, 100, 0.2)",
        line=dict(color="#00FF66", width=2),
    )
)

# 4. Espejos con su orientación física real
for idx, m in enumerate(mirrors):
  m_rad = np.radians(m["angle"])
  m_len = 3.0
  mx1 = m["x"] - np.cos(m_rad) * (m_len / 2)
  my1 = m["y"] - np.sin(m_rad) * (m_len / 2)
  mx2 = m["x"] + np.cos(m_rad) * (m_len / 2)
  my2 = m["y"] + np.sin(m_rad) * (m_len / 2)

  fig.add_trace(
      go.Scatter(
          x=[mx1, mx2],
          y=[my1, my2],
          mode="lines+markers",
          name=f"Espejo {idx+1}",
          line=dict(color="#00E5FF", width=7),
          marker=dict(size=4, color="white"),
      )
  )

# Ajustes de diseño de la interfaz visual
fig.update_layout(
    xaxis_title="Eje X (Lateral)",
    xaxis_range=[-max_x / 2 - 1, max_x / 2 + 1],
    xaxis_zeroline=True,
    yaxis_title="Eje Y (Fondo)",
    yaxis_range=[-1, max_y + 1],
    yaxis_scaleanchor="x",
    width=850,
    height=600,
    template="plotly_dark",
    legend=dict(x=0, y=1),
)

st.plotly_chart(fig, use_container_width=True)
