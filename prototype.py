import pyvista as pv
import numpy as np
import trimesh

# ============================================================
# LOADING THE MODEL
# ============================================================
MODEL_PATH = "lung_model.glb"

scene = trimesh.load(MODEL_PATH)
mesh_trimesh = trimesh.util.concatenate(tuple(scene.geometry.values()))

vertices_raw = mesh_trimesh.vertices.copy()
faces = mesh_trimesh.faces

faces_pv = np.hstack([[3, f[0], f[1], f[2]] for f in faces])
mesh = pv.PolyData(vertices_raw, faces_pv)

mesh = mesh.smooth(
    n_iter=30,
    relaxation_factor=0.015,
    feature_smoothing=False,
    boundary_smoothing=True,
)

vertices0 = mesh.points.copy()
mesh.compute_normals(inplace=True)
normals0 = mesh.point_normals.copy()
normals0 = normals0 / (np.linalg.norm(normals0, axis=1, keepdims=True) + 1e-8)

# ============================================================
# GEOMETRY SETUP
# ============================================================
center = vertices0.mean(axis=0)
coords = vertices0 - center

r = np.linalg.norm(coords, axis=1)
r_norm = (r - r.min()) / (r.max() - r.min() + 1e-8)

subpleural = np.clip((r_norm - 0.35) / 0.65, 0, 1)

z = coords[:, 2]
z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)
basal = 1.0 - z_norm

weight = 0.20 + 0.80 * (subpleural * basal)

# ============================================================
# STABLE DISEASE PATTERNS
# ============================================================
np.random.seed(42)

candidate_idx = np.where(weight > np.percentile(weight, 55))[0]
num_lesions = min(10, len(candidate_idx))
lesion_idx = np.random.choice(candidate_idx, num_lesions, replace=False)
lesion_centers = vertices0[lesion_idx]

base_radius = np.percentile(r, 9)

random_field = np.random.normal(0.5, 0.14, len(vertices0))
random_field = np.clip(random_field, 0, 1)

cluster_field = (
    np.sin(coords[:, 0] * 1.4 + coords[:, 1] * 0.6)
    + np.cos(coords[:, 1] * 1.2 - coords[:, 2] * 0.5)
)
cluster_field = (cluster_field - cluster_field.min()) / (
    cluster_field.max() - cluster_field.min() + 1e-8
)

# ============================================================
# SMOOTH HELPERS
# ============================================================
def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)

def lerp(a, b, x):
    return a * (1 - x) + b * x

def progression_params(t):
    """
    Smoothly interpolates disease behaviour.
    No sudden jump between stages.
    """
    t = np.clip(t, 0, 1)

    # Smooth severity curve
    s = smoothstep(t)

    # These are continuous, not sudden stage jumps
    radius_mult = lerp(0.75, 2.8, s)
    intensity_mult = lerp(0.10, 1.90, s)
    deformation_mult = lerp(0.00, 1.00, s)

    return radius_mult, intensity_mult, deformation_mult

# ============================================================
# DISEASE FIELD
# ============================================================
def compute_disease(t: float) -> np.ndarray:
    disease = np.zeros(len(vertices0))

    radius_mult, intensity_mult, _ = progression_params(t)
    radius = base_radius * radius_mult

    for lc in lesion_centers:
        dist = np.linalg.norm(vertices0 - lc, axis=1)
        influence = np.exp(-(dist ** 2) / (2 * radius ** 2))
        disease = np.maximum(disease, influence)

    # Stable organic variation
    # organic cloudy disease texture
    blob = (
        np.sin(coords[:, 0] * 1.2)
        + np.cos(coords[:, 1] * 1.4)
        + np.sin(coords[:, 2] * 1.1)
        )
    blob = (blob - blob.min()) / (blob.max() - blob.min() + 1e-8)
    disease *= (0.78 + 0.30 * random_field)
    disease *= (0.60 + 0.60 * blob)
    disease *= (0.75 + 0.35 * cluster_field)

    # Anatomical region weighting
    disease *= weight

    # Smooth intensity progression
    disease *= intensity_mult

    # Avoid tiny noise everywhere
    disease = np.where(disease > 0.025, disease, disease * 0.10)

    return np.clip(disease, 0, 1)

# ============================================================
# STRUCTURAL DEFORMATION
# ============================================================
def deform(t: float, disease: np.ndarray) -> np.ndarray:
    verts = vertices0.copy()

    _, _, deform_mult = progression_params(t)

    mask_threshold = 0.035
    local = np.clip((disease - mask_threshold) / (1.0 - mask_threshold), 0, 1)
    local = local ** 1.45

    # Controlled deformation — realistic, not broken mesh
    volume_loss_strength = 0.055 * deform_mult
    indentation_strength = 0.018 * deform_mult
    roughness_strength = 0.014 * deform_mult
    collapse_strength = 0.025 * deform_mult

    center_pull = (center - vertices0) * (local * volume_loss_strength)[:, None]
    indentation = normals0 * (local * indentation_strength)[:, None]

    rough_pattern = (
        np.sin(coords[:, 0] * 5.5 + coords[:, 1] * 1.5)
        + np.cos(coords[:, 1] * 4.8 - coords[:, 2] * 1.9)
        + np.sin(coords[:, 2] * 4.0)
    ) / 3.0

    roughness = normals0 * (rough_pattern * local * roughness_strength)[:, None]

    downward = np.zeros_like(vertices0)
    downward[:, 2] = -local * collapse_strength

    verts = verts + center_pull - indentation + roughness + downward
    return verts

# ============================================================
# REALISTIC TISSUE COLORS
# ============================================================
def make_tissue_colors(disease: np.ndarray, t: float) -> np.ndarray:
    healthy = np.array([235, 170, 175], dtype=float) / 255.0
    early = np.array([215, 120, 120], dtype=float) / 255.0
    moderate = np.array([160, 75, 75], dtype=float) / 255.0
    advanced = np.array([85, 32, 32], dtype=float) / 255.0

    d = np.clip((disease - 0.02) / 0.98, 0, 1)
    d = d ** 0.85

    s = smoothstep(t)

    base_color = lerp(healthy, early, smoothstep(t * 1.4))
    lesion_mid = lerp(early, moderate, smoothstep(t))
    lesion_deep = lerp(moderate, advanced, smoothstep((t - 0.45) / 0.55))
    lesion_color = lerp(lesion_mid, lesion_deep, smoothstep((t - 0.35) / 0.65))

    lesion_strength = lerp(0.10, 1.00, s)
    blend = np.clip(d * lesion_strength, 0, 1)

    colors = base_color[None, :] * (1 - blend[:, None]) + lesion_color[None, :] * blend[:, None]

    # Subtle tissue tonal variation
    tonal = (
        np.sin(coords[:, 0] * 2.0)
        + np.cos(coords[:, 1] * 1.5)
        + np.sin(coords[:, 2] * 1.1)
    )
    tonal = (tonal - tonal.min()) / (tonal.max() - tonal.min() + 1e-8)
    tonal = 0.86 + 0.22 * tonal
    colors *= tonal[:, None]

    # Slight wet/highlight effect
    highlight = np.clip(disease * 0.25, 0, 1)
    colors = colors + highlight[:, None] * np.array([0.08, 0.04, 0.04])

    return np.clip(colors * 255, 0, 255).astype(np.uint8)

# ============================================================
# TEXT LABELS
# ============================================================
def get_stage_label(t: float) -> str:
    if t < 0.20:
        return "Stage: Healthy"
    elif t < 0.45:
        return "Stage: Early Disease"
    elif t < 0.75:
        return "Stage: Moderate Disease"
    return "Stage: Advanced Disease"

def get_explanation(t: float) -> str:
    if t < 0.20:
        return "No major abnormalities. Smooth tissue appearance."
    elif t < 0.45:
        return "Early disease: small localized density changes begin to appear."
    elif t < 0.75:
        return "Moderate disease: abnormal regions spread with visible tissue deformation."
    return "Advanced disease: widespread density change, volume loss, and roughened morphology."

# ============================================================
# INITIAL STATE
# ============================================================
initial_t = 0.0
initial_disease = compute_disease(initial_t)

mesh.points = deform(initial_t, initial_disease)
mesh["disease"] = initial_disease
mesh["tissue_rgb"] = make_tissue_colors(initial_disease, initial_t)

# ============================================================
# PLOTTER
# ============================================================
plotter = pv.Plotter(window_size=[1250, 850])

plotter.remove_all_lights()

key = pv.Light(position=(3, -4, 3), focal_point=center, color="white", intensity=1.0)
fill = pv.Light(position=(-3, -2, 2), focal_point=center, color="white", intensity=0.4)
rim = pv.Light(position=(0, 3, 2), focal_point=center, color="white", intensity=0.6)

plotter.add_light(key)
plotter.add_light(fill)
plotter.add_light(rim)

plotter.set_background("white")

plotter.add_mesh(
    mesh,
    scalars="tissue_rgb",
    rgb=True,
    smooth_shading=True,
    specular=0.4,          # increase shine
    specular_power=25,     # sharper highlight
    ambient=0.15,
    diffuse=0.85,
)

plotter.add_axes()
plotter.enable_eye_dome_lighting()
plotter.enable_parallel_projection()

# Camera
plotter.reset_camera()
plotter.camera_position = [
    (0.0, -1.6, 0.7),
    tuple(center),
    (0, 0, 1),
]
plotter.camera.zoom(1.55)

plotter.add_text(
    get_stage_label(initial_t),
    position="upper_left",
    font_size=16,
    color="black",
    name="stage_label",
)

plotter.add_text(
    get_explanation(initial_t),
    position="lower_left",
    font_size=11,
    color="black",
    name="explanation_label",
)

# ============================================================
# UPDATE CALLBACK
# ============================================================
def update(val):
    t = float(val)

    disease = compute_disease(t)
    new_points = deform(t, disease)
    new_colors = make_tissue_colors(disease, t)

    mesh.points = new_points
    mesh["disease"] = disease
    mesh["tissue_rgb"] = new_colors

    plotter.add_text(
        get_stage_label(t),
        position="upper_left",
        font_size=16,
        color="black",
        name="stage_label",
    )

    plotter.add_text(
        get_explanation(t),
        position="lower_left",
        font_size=11,
        color="black",
        name="explanation_label",
    )

    plotter.render()

# ============================================================
# SLIDER
# ============================================================
plotter.add_slider_widget(
    update,
    [0.0, 1.0],
    value=0.0,
    title="Disease Progression",
    pointa=(0.35, 0.90),
    pointb=(0.85, 0.90),
)

plotter.show()
