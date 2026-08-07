import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Configuración de la interfaz en modo ancho
st.set_page_config(
    page_title="Simulador de Láser y Espejos", page_icon="💡", layout="wide"
)

st.title("💡 Simulador de Trayectoria Láser con Rebotes Reales")
st.markdown(
    "Visualiza los rebotes exactos de la luz usando la ley de reflexión"
    " vectorial ($\\theta_i = \\theta_r$)."
)

# --- PANEL DE CONTROL LATERAL (DERECHA) ---
st.sidebar.header("🎛️ CONTROLES (DERECHA)")

# 1. Dimensiones del Campo
st.sidebar.subheader("Dimensiones del Área")
max_x = st.sidebar.slider("Límite Eje X (Ancho)", 10.0, 50.0, 20.0, 1.0)
max_y = st.sidebar.slider("Límite Eje Y (Fondo)", 10.0, 50.0, 20.0, 1.0)

# 2. Configuración del Láser
st.sidebar.subheader("Láser (Origen)")
laser_x = st.sidebar.slider(
    "Posición X del Láser", -max_x / 2, max_x / 2, 0.0, 0.5
)
laser_angle_deg = st.sidebar.slider(
    "Ángulo del Láser (grados)", 0, 180, 90, 1
)

# 3. Configuración del Objetivo
st.sidebar.subheader("🎯 Objetivo")
target_x = st.sidebar.slider(
    "Posición X del Objetivo", -max_x / 2, max_x / 2, -5.0, 0.5
)
target_y = st.sidebar.slider(
    "Posición Y del Objetivo", 0.0, max_y, 5.0, 0.5
)
target_radius = st.sidebar.slider("Radio del Objetivo", 0.5, 3.0, 1.0, 0.2)

# 4. Configuración de Espejos (Mínimo 1, Máximo 5)
st.sidebar.subheader("🪞 Espejos")
num_mirrors = st.sidebar.slider("Cantidad de Espejos", 1, 5, 2)

mirrors = []
for i in range(num_mirrors):
  st.sidebar.markdown(f"**Espejo {i+1}**")
  mx = st.sidebar.slider(
      f"X Pos #{i+1}", -max_x / 2, max_x / 2, float(i * 6 - 3), 0.5, key=f"mx_{i}"
  )
  my = st.sidebar.slider(
      f"Y Pos #{i+1}", 0.0, max_y, float(i * 4 + 3), 0.5, key=f"my_{i}"
  )
  m_ang = st.sidebar.slider(
      f"Ángulo #{i+1} (grados)", 0, 180, 45, 1, key=f"mang_{i}"
  )
  mirrors.append({"x": mx, "y": my, "angle": m_ang})


# --- MOTOR DE FÍSICA Y REBOTES PRECISO ---
def get_ray_path(
    l_x, l_start_deg, mirrors_list, t_x, t_y, t_rad, b_x, b_y, max_bounces=15
):
  path_x = [l_x]
  path_y = [0.0]

  rad = np.radians(l_start_deg)
  dir_x = np.cos(rad)
  dir_y = np.sin(rad)
  curr_x, curr_y = l_x, 0.0

  hit_target = False
  bounce_points = []

  bound_x = b_x
  bound_y = b_y

  for _ in range(max_bounces):
    closest_t = float("inf")
    next_x, next_y = curr_x + dir_x * 1000, curr_y + dir_y * 1000
    normal_vector = None

    # 1. Comprobar colisión con las paredes de la caja
    if dir_x > 0:
      t_w = (bound_x - curr_x) / dir_x
      if 1e-3 < t_w < closest_t:
        closest_t = t_w
        next_x = bound_x
        next_y = curr_y + dir_x * t_w
        normal_vector = (-1, 0)
    elif dir_x < 0:
      t_w = (-bound_x - curr_x) / dir_x
      if 1e-3 < t_w < closest_t:
        closest_t = t_w
        next_x = -bound_x
        next_y = curr_y + dir_x * t_w
        normal_vector = (1, 0)

    if dir_y > 0:
      t_w = (bound_y - curr_y) / dir_y
      if 1e-3 < t_w < closest_t:
        closest_t = t_w
        next_x = curr_x + dir_x * t_w
        next_y = bound_y
        normal_vector = (0, -1)

    # 2. Comprobar colisión estricta con cada espejo (Segmento de línea)
    mirror_length = 4.0
    for idx, m in enumerate(mirrors_list):
      m_rad = np.radians(m["angle"])
      # Vector director del espejo
      m_dx = np.cos(m_rad) * (mirror_length / 2)
      m_dy = np.sin(m_rad) * (mirror_length / 2)

      x1, y1 = m["x"] - m_dx, m["y"] - m_dy
      x2, y2 = m["x"] + m_dx, m["y"] + m_dy

      # Sistema de ecuaciones paramétricas: Rayo vs Segmento de Espejo
      # Rayo: P = P0 + t * D
      # Espejo: Q = S1 + u * (S2 - S1)
      v1x = curr_x - x1
      v1y = curr_y - y1
      v2x = x2 - x1
      v2y = y2 - y1
      v3x = -dir_y
      v3y = dir_x

      dot = v2x * v3x + v2y * v3y
      if abs(dot) < 1e-6:
        continue

      t = (v2x * v1y - v2y * v1x) / dot
      u = (v1x * v3x + v1y * v3y) / dot

      # t > 1e-3 previene que el láser detecte el punto del que acaba de rebotar
      # u entre 0 y 1 asegura que golpee estrictamente dentro de la barra del espejo
      if 1e-3 < t < closest_t and 0.0 <= u <= 1.0:
        closest_t = t
        next_x = curr_x + dir_x * t
        next_y = curr_y + dir_y * t

        # Vector normal perpendicular a la superficie del espejo
        nx, ny = -v2y, v2x
        length_n = np.hypot(nx, ny)
        if length_n > 0:
          nx, ny = nx / length_n, ny / length_n

        # Asegurar que la normal apunte en sentido opuesto al rayo incidente
        if nx * dir_x + ny * dir_y > 0:
          nx, ny = -nx, -ny

        normal_vector = (nx, ny)

    # 3. Comprobar si el objetivo interseca esta sección del rayo
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

    if normal_vector is None:
      break

    # Registrar rebote real
    bounce_points.append((next_x, next_y))

    # Ley de reflexión óptica exacta: R = D - 2(D · N) * N
    d_vec = np.array([dir_x, dir_y])
    n_vec = np.array(normal_vector)
    r_vec = d_vec - 2 * np.dot(d_vec, n_vec) * n_vec
    dir_x, dir_y = r_vec[0], r_vec[1]
    curr_x, curr_y = next_x, next_y

  return path_x, path_y, hit_target, bounce_points


# Ejecutar la simulación con la física corregida
path_x, path_y, success, bounces = get_ray_path(
    laser_x,
    laser_angle_deg,
    mirrors,
    target_x,
    target_y,
    target_radius,
    max_x / 2,
    max_y,
)

# --- PANEL DE INFORMACIÓN INFERIOR ---
col1, col2 = st.columns(2)
with col1:
  if success:
    st.success(
        f"🎯 ¡Impacto exitoso! El láser alcanzó el objetivo tras {len(bounces)}"
        " rebote(s)."
    )
  else:
    st.warning(
        "⚠️ El láser no ha alcanzado el objetivo. Modifica los ángulos y"
        " posiciones."
    )
with col2:
  st.info(f"🔄 Rebotes totales realizados: **{len(bounces)}**")


# --- CONSTRUCCIÓN DE LA GRÁFICA INTERACTIVA CON PLOTLY ---
fig = go.Figure()

# 1. Trazado principal del rayo láser
fig.add_trace(
    go.Scatter(
        x=path_x,
        y=path_y,
        mode="lines+markers",
        name="Rayo Láser",
        line=dict(color="#00FF66", width=3),
        marker=dict(size=5, color="#00FF66"),
    )
)

# 2. Marcadores específicos en los puntos de rebote
if bounces:
  bx_vals = [b[0] for b in bounces]
  by_vals = [b[1] for b in bounces]
  fig.add_trace(
      go.Scatter(
          x=bx_vals,
          y=by_vals,
          mode="markers+text",
          name="Puntos de Rebote",
          marker=dict(size=14, color="yellow", symbol="diamond"),
          text=[f"Rebote {i+1}" for i in range(len(bounces))],
          textposition="top center",
          textfont=dict(color="yellow"),
      )
  )

# 3. Emisor Láser en el eje X (Y=0)
fig.add_trace(
    go.Scatter(
        x=[laser_x],
        y=[0],
        mode="markers+text",
        name="Láser",
        marker=dict(color="#FF00FF", size=16, symbol="triangle-up"),
        text=["LÁSER"],
        textposition="bottom center",
        textfont=dict(color="#FF00FF"),
    )
)

# 4. Objetivo circular verde
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
        fillcolor="rgba(0, 255, 100, 0.25)",
        line=dict(color="#00FF66", width=2),
    )
)
fig.add_trace(
    go.Scatter(
        x=[target_x],
        y=[target_y],
        mode="markers",
        showlegend=False,
        marker=dict(color="#00FF66", size=8),
    )
)

# 5. Espejos configurables individualmente
for idx, m in enumerate(mirrors):
  m_rad = np.radians(m["angle"])
  m_len = 4.0
  mx1 = m["x"] - np.cos(m_rad) * (m_len / 2)
  my1 = m["y"] - np.sin(m_rad) * (m_len / 2)
  mx2 = m["x"] + np.cos(m_rad) * (m_len / 2)
  my2 = m["y"] + np.sin(m_rad) * (m_len / 2)

  fig.add_trace(
      go.Scatter(
          x=[mx1, mx2],
          y=[my1, my2],
          mode="lines+markers+text",
          name=f"Espejo {idx+1}",
          line=dict(color="#00E5FF", width=8),
          marker=dict(size=6, color="white"),
          text=[f"Espejo {idx+1}", ""],
          textposition="top center",
          textfont=dict(color="#00E5FF"),
      )
  )

# Propiedades del diseño visual del plano cartesiano
fig.update_layout(
    xaxis_title="Eje X (Lateral)",
    xaxis_range=[-max_x / 2 - 2, max_x / 2 + 2],
    xaxis_zeroline=True,
    yaxis_title="Eje Y (Fondo)",
    yaxis_range=[-2, max_y + 2],
    yaxis_scaleanchor="x",
    width=900,
    height=650,
    template="plotly_dark",
    legend=dict(x=0, y=1),
)

st.plotly_chart(fig, use_container_width=True)
