"""
Generador de Mapas Topográficos 3D para Impresión
--------------------------------------------------
Genera modelos STL listos para impresora 3D con:
  • Terreno real (SRTM 90 m via OpenTopoData)
  • Edificios de OpenStreetMap
  • Rutas GPX como relieve alzado
"""

import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.buildings import fetch_buildings
from src.gpx_handler import get_gpx_bounds, parse_gpx
from src.mesh_generator import MeshGenerator
from src.terrain import fetch_elevation_grid

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Generador de Mapas 3D",
    page_icon="🗺️",
    layout="wide",
)

st.title("🗺️ Generador de Mapas Topográficos 3D")
st.caption(
    "Terreno real · Edificios OSM · Rutas GPX · Exporta STL para impresión 3D"
)

# ─────────────────────────────────────────────────────────────
# Sidebar – Configuration
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuración")

    # ── Location ──────────────────────────────────────────────
    st.subheader("📍 Ubicación")

    gpx_file = st.file_uploader(
        "Cargar archivo GPX (opcional)",
        type=["gpx"],
        help="Si cargas un GPX primero, puedes centrar el mapa en la ruta.",
    )

    gpx_points = None
    gpx_bounds = None
    if gpx_file is not None:
        try:
            gpx_points = parse_gpx(gpx_file.read())
            gpx_bounds = get_gpx_bounds(gpx_points)
            if gpx_bounds:
                st.success(f"✅ {len(gpx_points)} puntos GPX cargados")
        except Exception as exc:
            st.error(f"Error leyendo GPX: {exc}")

    default_lat = gpx_bounds["lat_center"] if gpx_bounds else 40.4168
    default_lon = gpx_bounds["lon_center"] if gpx_bounds else -3.7038
    default_size = min(gpx_bounds["size_km"], 50.0) if gpx_bounds else 5.0

    lat_center = st.number_input(
        "Latitud central", value=default_lat, format="%.6f"
    )
    lon_center = st.number_input(
        "Longitud central", value=default_lon, format="%.6f"
    )
    size_km = st.slider(
        "Tamaño del área (km)", min_value=1.0, max_value=50.0,
        value=float(f"{default_size:.1f}"), step=0.5,
    )

    # ── 3D model ──────────────────────────────────────────────
    st.subheader("🖨️ Modelo 3D")
    target_size_mm  = st.slider("Tamaño del modelo (mm)", 50,  300, 150, 10)
    base_thickness  = st.slider("Grosor de la base (mm)",  1.0, 10.0,  3.0, 0.5)
    max_ele_height  = st.slider("Altura máx. relieve (mm)", 5.0, 50.0, 20.0, 1.0)

    res_map   = {"Baja (30×30)": 30, "Media (50×50)": 50, "Alta (80×80)": 80}
    res_label = st.selectbox("Resolución del terreno", list(res_map.keys()), index=1)
    resolution = res_map[res_label]

    # ── Buildings ─────────────────────────────────────────────
    st.subheader("🏢 Edificios")
    include_buildings = st.checkbox(
        "Incluir edificios (OSM)",
        value=False,
        help="Recomendado solo para áreas < 5 km.",
    )
    building_height = st.slider("Altura edificios (mm)", 0.5, 10.0, 2.0, 0.5) \
        if include_buildings else 2.0

    # ── Route ─────────────────────────────────────────────────
    if gpx_points:
        st.subheader("🛤️ Ruta GPX")
        route_width  = st.slider("Ancho de la ruta (mm)",  0.5, 5.0, 1.0, 0.25)
        route_height = st.slider("Altura de la ruta (mm)", 0.5, 5.0, 1.0, 0.25)
    else:
        route_width, route_height = 1.0, 1.0

# ─────────────────────────────────────────────────────────────
# Helper: compute bounds from center + size
# ─────────────────────────────────────────────────────────────
lat_delta = (size_km / 2.0) / 111.0
lon_delta = (size_km / 2.0) / (111.0 * math.cos(math.radians(lat_center)))
lat_min_view = lat_center - lat_delta
lat_max_view = lat_center + lat_delta
lon_min_view = lon_center - lon_delta
lon_max_view = lon_center + lon_delta

# ─────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────
tab_map, tab_3d, tab_stl = st.tabs(
    ["🗺️ Vista del Área", "🏔️ Preview 3D", "⬇️ Generar STL"]
)

# ══════════════════════════════════════════════════════════════
# TAB 1 – 2D Map preview
# ══════════════════════════════════════════════════════════════
with tab_map:
    st.subheader("Área seleccionada")

    fig = go.Figure()

    # Selection rectangle
    rect_lats = [lat_min_view, lat_max_view, lat_max_view, lat_min_view, lat_min_view]
    rect_lons = [lon_min_view, lon_min_view, lon_max_view, lon_max_view, lon_min_view]
    fig.add_trace(go.Scattermapbox(
        lat=rect_lats, lon=rect_lons,
        mode="lines",
        line=dict(color="red", width=2),
        name="Área",
    ))
    fig.add_trace(go.Scattermapbox(
        lat=[lat_center], lon=[lon_center],
        mode="markers",
        marker=dict(size=10, color="red"),
        name="Centro",
    ))

    if gpx_points:
        fig.add_trace(go.Scattermapbox(
            lat=[p[0] for p in gpx_points],
            lon=[p[1] for p in gpx_points],
            mode="lines",
            line=dict(color="blue", width=3),
            name="Ruta GPX",
        ))

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=lat_center, lon=lon_center),
            zoom=10,
        ),
        margin=dict(r=0, t=0, l=0, b=0),
        height=500,
        legend=dict(x=0.01, y=0.99),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"Área: {size_km} km × {size_km} km  |  "
        f"Centro: {lat_center:.4f}°, {lon_center:.4f}°  |  "
        f"Bounds: [{lat_min_view:.4f}, {lat_max_view:.4f}] lat · "
        f"[{lon_min_view:.4f}, {lon_max_view:.4f}] lon"
    )

# ══════════════════════════════════════════════════════════════
# TAB 2 – 3D preview
# ══════════════════════════════════════════════════════════════
with tab_3d:
    st.subheader("Vista 3D del terreno")

    est_calls = math.ceil(resolution * resolution / 100)
    est_secs  = math.ceil(est_calls * 1.2)
    st.info(
        f"Resolución {resolution}×{resolution} = {resolution**2} puntos "
        f"→ ~{est_calls} llamadas API → ≈ {est_secs} s de descarga."
    )

    if st.button("🔄 Cargar elevaciones y mostrar preview"):
        with st.spinner("Descargando datos de elevación…"):
            try:
                grid, lat_b, lon_b = fetch_elevation_grid(
                    lat_center, lon_center, size_km, resolution
                )
                st.session_state["grid"]     = grid
                st.session_state["lat_b"]    = lat_b
                st.session_state["lon_b"]    = lon_b
                st.success(
                    f"Elevación: {grid.min():.0f} m – {grid.max():.0f} m  "
                    f"(rango {grid.max()-grid.min():.0f} m)"
                )
            except Exception as exc:
                st.error(f"Error descargando elevaciones: {exc}")

    if "grid" in st.session_state:
        grid  = st.session_state["grid"]
        lat_b = st.session_state["lat_b"]
        lon_b = st.session_state["lon_b"]

        rows, cols = grid.shape
        xs = np.linspace(lon_b[0], lon_b[1], cols)
        ys = np.linspace(lat_b[0], lat_b[1], rows)

        fig3d = go.Figure()
        fig3d.add_trace(go.Surface(
            z=grid, x=xs, y=ys,
            colorscale="Earth",
            showscale=True,
            colorbar=dict(title="Elevación (m)"),
        ))

        if gpx_points:
            gen = MeshGenerator()
            gen._setup(grid, lat_b, lon_b)
            in_area = [
                p for p in gpx_points
                if lat_b[0] <= p[0] <= lat_b[1] and lon_b[0] <= p[1] <= lon_b[1]
            ]
            if in_area:
                rz = [
                    float(gen._interp([[p[0], p[1]]])[0]) + 100
                    for p in in_area
                ]
                fig3d.add_trace(go.Scatter3d(
                    x=[p[1] for p in in_area],
                    y=[p[0] for p in in_area],
                    z=rz,
                    mode="lines",
                    line=dict(color="royalblue", width=4),
                    name="Ruta GPX",
                ))

        fig3d.update_layout(
            height=620,
            scene=dict(
                xaxis_title="Longitud",
                yaxis_title="Latitud",
                zaxis_title="Elevación (m)",
                aspectmode="manual",
                aspectratio=dict(x=1, y=1, z=0.25),
            ),
            margin=dict(r=0, t=20, l=0, b=0),
        )
        st.plotly_chart(fig3d, use_container_width=True)
    else:
        st.info("Pulsa el botón de arriba para cargar los datos y ver el preview 3D.")

# ══════════════════════════════════════════════════════════════
# TAB 3 – Generate STL
# ══════════════════════════════════════════════════════════════
with tab_stl:
    st.subheader("Generar modelo STL para impresión 3D")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Resumen de parámetros:**")
        st.write(f"- Centro: {lat_center:.5f}°N, {lon_center:.5f}°E")
        st.write(f"- Área: {size_km} km × {size_km} km")
        st.write(f"- Tamaño modelo: {target_size_mm} mm")
        st.write(f"- Base: {base_thickness} mm · Relieve: {max_ele_height} mm")
        st.write(f"- Resolución: {resolution}×{resolution}")
        st.write(f"- Edificios: {'Sí' if include_buildings else 'No'}")
        bld_warn = include_buildings and size_km > 5
        if bld_warn:
            st.warning("⚠️ Área > 5 km con edificios puede tardar mucho.")
        gpx_label = (
            f"Sí ({len(gpx_points)} puntos)" if gpx_points else "No"
        )
        st.write(f"- Ruta GPX: {gpx_label}")

    with col_r:
        st.markdown("**Tiempo estimado:**")
        est_calls = math.ceil(resolution * resolution / 100)
        est_secs  = math.ceil(est_calls * 1.2)
        st.metric("Descarga elevación", f"≈ {est_secs} s")
        if include_buildings:
            st.metric("Descarga edificios", "≈ 10–30 s")
        st.metric("Generación malla", "< 5 s")

    st.divider()

    if st.button("🚀 Generar STL", type="primary"):
        config = {
            "target_size_mm":    target_size_mm,
            "base_thickness_mm": base_thickness,
            "max_ele_height_mm": max_ele_height,
            "building_height_mm": building_height,
            "route_width_mm":    route_width,
            "route_height_mm":   route_height,
        }

        # 1. Elevation
        progress = st.progress(0, text="Descargando elevaciones…")
        try:
            grid, lat_b, lon_b = fetch_elevation_grid(
                lat_center, lon_center, size_km, resolution
            )
            st.session_state["grid"]  = grid
            st.session_state["lat_b"] = lat_b
            st.session_state["lon_b"] = lon_b
            progress.progress(40, text=f"Elevación OK ({grid.min():.0f}–{grid.max():.0f} m)")
        except Exception as exc:
            st.error(f"Error descargando elevaciones: {exc}")
            st.stop()

        # 2. Buildings (optional)
        buildings_data = None
        if include_buildings:
            progress.progress(45, text="Descargando edificios…")
            try:
                buildings_data = fetch_buildings(lat_b[0], lat_b[1], lon_b[0], lon_b[1])
                progress.progress(70, text=f"{len(buildings_data)} edificios descargados")
            except Exception as exc:
                st.warning(f"No se pudieron cargar edificios: {exc}")
        else:
            progress.progress(70, text="Edificios: omitidos")

        # 3. Generate mesh
        progress.progress(75, text="Generando malla 3D…")
        try:
            gen = MeshGenerator(config)
            stl_bytes = gen.generate_bytes(
                grid, lat_b, lon_b,
                buildings=buildings_data,
                gpx_points=gpx_points,
            )
            progress.progress(100, text="✅ STL generado")
            kb = len(stl_bytes) / 1024
            st.success(f"Modelo listo: {kb:.1f} KB")
        except Exception as exc:
            st.error(f"Error generando malla: {exc}")
            import traceback
            st.code(traceback.format_exc())
            st.stop()

        filename = f"mapa3d_{lat_center:.3f}_{lon_center:.3f}_{size_km}km.stl"
        st.download_button(
            label="⬇️ Descargar STL",
            data=stl_bytes,
            file_name=filename,
            mime="application/octet-stream",
        )

        st.markdown(
            "**Instrucciones de impresión recomendadas:**\n"
            "- Relleno: 15–20 %\n"
            "- Capa: 0.15–0.20 mm\n"
            "- Sin soportes necesarios\n"
            "- Material: PLA o PETG\n"
        )

# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Datos: [OpenTopoData](https://www.opentopodata.org/) (SRTM 90m) · "
    "[OpenStreetMap](https://www.openstreetmap.org/) · "
    "GPX propio"
)
