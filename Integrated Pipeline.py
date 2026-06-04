import argparse
import os
import json
import uuid
import datetime
import cv2
import numpy as np
import pyvista as pv
import trimesh
import matplotlib.pyplot as plt



parser = argparse.ArgumentParser()
parser.add_argument("ct_image", help="Path to CT image, example: case_08_image.jpg")
parser.add_argument("--model", default="lung_model.glb", help="Path to lung GLB model")
parser.add_argument("--show_ct", action="store_true", help="Show CT segmentation output")
parser.add_argument("--save_audit", action="store_true", help="Save audit log JSON on exit")
args = parser.parse_args()

CT_IMAGE_PATH = args.ct_image
MODEL_PATH = args.model




class ProgressionAuditLogger:
    """
    Records the deterministic mapping from CT imaging signals
    to 3D mesh modifications and exports a JSON audit trail.
    """

    def __init__(self, patient_id: str, output_dir: str = "audit_logs"):
        self.session_id  = str(uuid.uuid4())
        self.patient_id  = patient_id
        self.output_dir  = output_dir
        self.audit_trail = {
            "session_metadata": {
                "session_id": self.session_id,
                "patient_id": self.patient_id,
                "start_time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "pipeline":   "Organ Disease Progression Visualiser — COMP8851",
            },
            "events": [],
        }
        os.makedirs(self.output_dir, exist_ok=True)

    def _record_event(self, stage: str, data: dict):
        event = {
            "event_id":       str(uuid.uuid4()),
            "timestamp":      datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "pipeline_stage": stage,
            "event_data":     data,
        }
        self.audit_trail["events"].append(event)

    def log_ct_extraction(self, image_path, severity, texture, hdr, pixel_count):
        """Stage 1 — raw signals extracted live from the CT image."""
        self._record_event("1_CT_EXTRACTION", {
            "image_path":        image_path,
            "severity":          round(severity,    4),
            "texture":           round(texture,     4),
            "hdr":               round(hdr,         4),
            "lung_pixel_count":  pixel_count,
            "extraction_method": "Otsu_threshold + connected_components",
        })

    def log_feature_mapping(self, case_params):
        """Stage 2 — normalised signals and derived visual parameters."""
        self._record_event("2_FEATURE_MAPPING", {
            "severity_norm":        round(case_params["severity_norm"],       3),
            "texture_norm":         round(case_params["texture_norm"],        3),
            "hdr_norm":             round(case_params["hdr_norm"],            3),
            "progression_score":    round(case_params["progression_score"],   3),
            "weighting":            "0.65*severity + 0.20*texture + 0.15*hdr",
            "derived_params": {
                "colour_strength":      round(case_params["colour_strength"],      3),
                "roughness_strength":   round(case_params["roughness_strength"],   3),
                "lesion_spread":        round(case_params["lesion_spread"],        3),
                "deformation_strength": round(case_params["deformation_strength"], 3),
            },
        })

    def log_stage_transition(self, t, stage, radius_mult, intensity_mult, deform_mult):
        """Stage 3 — records when disease stage changes as slider moves."""
        self._record_event("3_STAGE_TRANSITION", {
            "slider_t":       round(t, 3),
            "disease_stage":  stage,
            "active_effects": {
                "lesion_radius_multiplier": round(radius_mult,    3),
                "intensity_multiplier":     round(intensity_mult, 3),
                "deformation_multiplier":   round(deform_mult,    3),
            },
            "clinical_justification": (
                f"CT signals drive mesh at t={t:.2f}: lesion radius {radius_mult:.2f}x, "
                f"tissue colour {intensity_mult:.2f}x, structural deformation {deform_mult:.2f}."
            ),
        })

    def log_vertex_modification(self, t, affected_count, vol_loss, roughness, collapse):
        """Stage 4 — mesh geometry changes applied to the 3D model."""
        self._record_event("4_VERTEX_MODIFICATION", {
            "slider_t":               round(t, 3),
            "affected_vertex_count":  affected_count,
            "deformation_components": {
                "volume_loss_strength": round(vol_loss,  5),
                "roughness_strength":   round(roughness, 5),
                "basal_collapse":       round(collapse,  5),
            },
            "mesh_region":           "subpleural_basal_weighted",
            "clinical_justification": (
                "Vertex shifts represent fibrotic contraction (centre-pull), "
                "pleural surface indentation, and basal collapse — "
                "consistent with IPF morphology."
            ),
        })

    def save_audit_log(self):
        patient_label = os.path.splitext(os.path.basename(CT_IMAGE_PATH))[0]
        filename = f"audit_log_{patient_label}_{self.session_id[:8]}.json"
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w") as f:
            json.dump(self.audit_trail, f, indent=4)
        print(f"\n[AUDIT] Log saved -> {filepath}")
        return filepath


# Initialise logger (always created, only saved if --save_audit is passed)
audit = ProgressionAuditLogger(patient_id=os.path.basename(CT_IMAGE_PATH))

def extract_ct_features(image_path, show=False):
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Could not load CT image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        thresh, connectivity=8
    )

    mask = np.zeros_like(gray)

    h, w = gray.shape
    image_area = h * w

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        bw = stats[label, cv2.CC_STAT_WIDTH]
        bh = stats[label, cv2.CC_STAT_HEIGHT]
        cx, cy = centroids[label]

        if area < 300:
            continue
        if area > 0.25 * image_area:
            continue
        if x <= 5 or y <= 5 or (x + bw) >= w - 5 or (y + bh) >= h - 5:
            continue
        if cy < 0.15 * h or cy > 0.85 * h:
            continue
        if cx < 0.10 * w or cx > 0.90 * w:
            continue

        mask[labels == label] = 255

    kernel_close = np.ones((7, 7), np.uint8)
    kernel_open = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    num_labels2, labels2, stats2, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    if num_labels2 > 1:
        component_areas = []

        for label in range(1, num_labels2):
            area = stats2[label, cv2.CC_STAT_AREA]
            component_areas.append((label, area))

        component_areas = sorted(component_areas, key=lambda x: x[1], reverse=True)

        final_mask = np.zeros_like(mask)

        for label, area in component_areas[:2]:
            final_mask[labels2 == label] = 255

        mask = final_mask

    lung_only = cv2.bitwise_and(gray, gray, mask=mask)

    disease_map = lung_only / 255.0
    lung_pixels = disease_map[mask > 0]

    if len(lung_pixels) == 0:
        raise ValueError("Segmentation failed: no lung pixels were found.")

    severity = float(np.mean(lung_pixels))
    texture = float(np.std(lung_pixels))
    hdr = float(np.sum(lung_pixels > 0.6) / len(lung_pixels))

    if show:
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.title("Cleaned CT")
        plt.imshow(gray, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Lung Mask")
        plt.imshow(mask, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Lung Region Only")
        plt.imshow(lung_only, cmap="gray")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    return {
        "severity":    severity,
        "texture":     texture,
        "hdr":         hdr,
        "pixel_count": int(len(lung_pixels)),
        "gray":        gray,
        "mask":        mask,
        "lung_only":   lung_only,
    }



def clamp01(x):
    return float(np.clip(x, 0, 1))


def normalise(value, min_value, max_value):
    return clamp01((value - min_value) / (max_value - min_value + 1e-8))


def map_features_to_parameters(features):
    severity = features["severity"]
    texture = features["texture"]
    hdr = features["hdr"]

    severity_norm = normalise(severity, 0.15, 0.50)
    texture_norm = normalise(texture, 0.07, 0.18)
    hdr_norm = normalise(hdr, 0.005, 0.085)

    progression_score = (
        0.65 * severity_norm +
        0.20 * texture_norm +
        0.15 * hdr_norm
    )

    return {
        "severity": severity,
        "texture": texture,
        "hdr": hdr,
        "severity_norm": severity_norm,
        "texture_norm": texture_norm,
        "hdr_norm": hdr_norm,
        "progression_score": progression_score,
        "colour_strength": 0.35 + 0.85 * severity_norm,
        "roughness_strength": 0.45 + 1.10 * texture_norm,
        "lesion_spread": 0.60 + 1.80 * hdr_norm,
        "deformation_strength": 0.25 + 0.90 * progression_score,
    }


features = extract_ct_features(CT_IMAGE_PATH, show=args.show_ct)
case_params = map_features_to_parameters(features)

# Log CT extraction and feature mapping to audit trail
audit.log_ct_extraction(
    image_path  = CT_IMAGE_PATH,
    severity    = features["severity"],
    texture     = features["texture"],
    hdr         = features["hdr"],
    pixel_count = features["pixel_count"],
)
audit.log_feature_mapping(case_params)

print("\n========== CT FEATURE EXTRACTION ==========")
print("Input CT image:", CT_IMAGE_PATH)
print("Severity:", round(case_params["severity"], 4))
print("Texture:", round(case_params["texture"], 4))
print("High Density Ratio:", round(case_params["hdr"], 4))
print("Progression score:", round(case_params["progression_score"], 3))
print("===========================================\n")




if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Could not find model file: {MODEL_PATH}")

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
# DISEASE PATTERN SETUP
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




def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def lerp(a, b, x):
    return a * (1 - x) + b * x


def progression_params(t):
    t = np.clip(t, 0, 1)
    s = smoothstep(t)

    radius_mult = lerp(
        0.75,
        2.8 * case_params["lesion_spread"],
        s
    )

    intensity_mult = lerp(
        0.10,
        1.90 * case_params["colour_strength"],
        s
    )

    deformation_mult = lerp(
        0.00,
        1.00 * case_params["deformation_strength"],
        s
    )

    return radius_mult, intensity_mult, deformation_mult




def compute_disease(t):
    disease = np.zeros(len(vertices0))

    radius_mult, intensity_mult, _ = progression_params(t)
    radius = base_radius * radius_mult

    for lc in lesion_centers:
        dist = np.linalg.norm(vertices0 - lc, axis=1)
        influence = np.exp(-(dist ** 2) / (2 * radius ** 2))
        disease = np.maximum(disease, influence)

    blob = (
        np.sin(coords[:, 0] * 1.2)
        + np.cos(coords[:, 1] * 1.4)
        + np.sin(coords[:, 2] * 1.1)
    )

    blob = (blob - blob.min()) / (blob.max() - blob.min() + 1e-8)

    disease *= (0.78 + 0.30 * random_field)
    disease *= (0.60 + 0.60 * blob)
    disease *= (0.75 + 0.35 * cluster_field)
    disease *= weight
    disease *= intensity_mult

    disease = disease ** 0.7

    disease = np.where(disease > 0.025, disease, disease * 0.10)

    return np.clip(disease, 0, 1)


# ============================================================
# STRUCTURAL DEFORMATION
# ============================================================

def deform(t, disease):
    verts = vertices0.copy()

    _, _, deform_mult = progression_params(t)

    mask_threshold = 0.035
    local = np.clip((disease - mask_threshold) / (1.0 - mask_threshold), 0, 1)
    local = local ** 1.45

    volume_loss_strength = 0.055 * deform_mult
    indentation_strength = 0.018 * deform_mult
    roughness_strength = 0.012 * deform_mult * case_params["roughness_strength"]
    collapse_strength = 0.030 * deform_mult

    center_pull = (center - vertices0) * (local * volume_loss_strength)[:, None]
    indentation = normals0 * (local * indentation_strength)[:, None]

    rough_pattern = (
        np.sin(coords[:, 0] * 5.5 + coords[:, 1] * 1.5)
        + np.cos(coords[:, 1] * 4.8 - coords[:, 2] * 1.9)
        + np.sin(coords[:, 2] * 4.0)
    ) / 3.0

    roughness = normals0 * (rough_pattern * local * roughness_strength)[:, None]

    downward = np.zeros_like(vertices0)
    downward[:, 2] = -local * collapse_strength * basal

    verts = verts + center_pull - indentation + roughness + downward

    return verts


# ============================================================
# TISSUE COLOURS
# ============================================================

def make_tissue_colors(disease, t):
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

    colors = (
        base_color[None, :] * (1 - blend[:, None])
        + lesion_color[None, :] * blend[:, None]
    )

    tonal = (
        np.sin(coords[:, 0] * 2.0)
        + np.cos(coords[:, 1] * 1.5)
        + np.sin(coords[:, 2] * 1.1)
    )

    tonal = (tonal - tonal.min()) / (tonal.max() - tonal.min() + 1e-8)
    tonal = 0.86 + 0.22 * tonal

    colors *= tonal[:, None]

    highlight = np.clip(disease * 0.25, 0, 1)
    colors = colors + highlight[:, None] * np.array([0.08, 0.04, 0.04])

    return np.clip(colors * 255, 0, 255).astype(np.uint8)


# ============================================================
# TEXT LABELS & LIVE EXPLANATION PANEL
# ============================================================

def get_stage_label(t):
    score = case_params["progression_score"]
    effective_stage = np.clip((0.60 * t) + (0.40 * score), 0, 1)

    if effective_stage < 0.20:
        return "Stage: Healthy / Minimal Abnormality"
    elif effective_stage < 0.45:
        return "Stage: Early Disease"
    elif effective_stage < 0.75:
        return "Stage: Moderate Disease"
    return "Stage: Advanced Disease"


def get_dominant_signal():
    sigs = {
        "Severity": case_params["severity_norm"],
        "Texture":  case_params["texture_norm"],
        "HDR":      case_params["hdr_norm"],
    }
    dom   = max(sigs, key=sigs.get)
    lvl   = sigs[dom]
    strength = "low" if lvl < 0.35 else ("moderate" if lvl < 0.65 else "high")
    return f"{dom}  ({strength},  norm={lvl:.2f})"


def get_active_effects(t):
    rm, im, dm = progression_params(t)
    p  = case_params
    sp = int((rm / (2.8 * p["lesion_spread"]))   * 100)
    cp = int((im / (1.90 * p["colour_strength"])) * 100)

    if dm < 0.05:   ds = "inactive"
    elif dm < 0.40: ds = f"mild       ({dm:.2f})"
    elif dm < 0.75: ds = f"moderate   ({dm:.2f})"
    else:           ds = f"strong     ({dm:.2f})"

    rn = 0.012 * dm * p["roughness_strength"]
    if rn < 0.003:   rs = "smooth"
    elif rn < 0.010: rs = f"mild fibrotic texture  ({rn:.4f})"
    else:            rs = f"pronounced fibrosis    ({rn:.4f})"

    return (
        f"  Lesion radius     :  {rm:.2f}x base  ({sp}% of case max)\n"
        f"  Tissue darkening  :  {im:.2f}x       ({cp}% of case max)\n"
        f"  Structural deform :  {ds}\n"
        f"  Surface roughness :  {rs}"
    )


def get_clinical_reasoning():
    p     = case_params
    score = p["progression_score"]
    sd = "low"     if p["severity_norm"] < 0.35 else ("elevated"  if p["severity_norm"] < 0.70 else "high")
    td = "regular" if p["texture_norm"]  < 0.35 else ("irregular" if p["texture_norm"]  < 0.70 else "highly irregular")
    hd = "minimal" if p["hdr_norm"]      < 0.35 else ("notable"   if p["hdr_norm"]      < 0.70 else "significant")

    if score < 0.25:   ov = "Near-healthy or very mild abnormality."
    elif score < 0.45: ov = "Early-stage disease with localised changes."
    elif score < 0.65: ov = "Moderate disease with spreading involvement."
    else:              ov = "Advanced disease with extensive involvement."

    return (
        f"  {ov}\n"
        f"  Mean intensity {p['severity']:.3f} ({sd}),\n"
        f"  tissue variation {p['texture']:.3f} ({td}),\n"
        f"  high-density ratio {p['hdr']:.3f} ({hd})."
    )


def build_explanation_panel(t):
    p   = case_params
    div = "-" * 46
    return (
        f"CT SOURCE: {os.path.basename(CT_IMAGE_PATH)}\n"
        f"{div}\n"
        f"RAW CT SIGNALS\n"
        f"  Severity  (mean intensity) : {p['severity']:.4f}  [norm {p['severity_norm']:.2f}]\n"
        f"  Texture   (std deviation)  : {p['texture']:.4f}  [norm {p['texture_norm']:.2f}]\n"
        f"  HDR       (density ratio)  : {p['hdr']:.4f}  [norm {p['hdr_norm']:.2f}]\n"
        f"  Progression score          : {p['progression_score']:.3f}"
        f"  (0.65*sev + 0.20*tex + 0.15*hdr)\n"
        f"{div}\n"
        f"DOMINANT DRIVER: {get_dominant_signal()}\n"
        f"{div}\n"
        f"ACTIVE EFFECTS  [t = {t:.2f}]\n"
        f"{get_active_effects(t)}\n"
        f"{div}\n"
        f"CLINICAL REASONING\n"
        f"{get_clinical_reasoning()}"
    )


def get_mapping_text():
    return (
        "Mapping: severity -> colour/deformation | "
        "texture -> surface roughness | "
        "HDR -> lesion spread"
    )


# Audit: track last logged stage to avoid duplicate log entries
_last_logged_stage = [None]

def maybe_log_stage(t):
    stage = get_stage_label(t)
    if stage == _last_logged_stage[0]:
        return
    _last_logged_stage[0] = stage

    rm, im, dm = progression_params(t)
    affected = int(np.sum(
        np.clip((compute_disease(t) - 0.015) / (1.0 - 0.015), 0, 1) ** 1.25 > 0.01
    ))
    audit.log_stage_transition(t, stage, rm, im, dm)
    audit.log_vertex_modification(
        t            = t,
        affected_count = affected,
        vol_loss     = 0.16 * dm,
        roughness    = 0.012 * dm * case_params["roughness_strength"],
        collapse     = 0.09 * dm,
    )


# ============================================================
# INITIAL STATE
# ============================================================

current_t = case_params["progression_score"]

initial_disease = compute_disease(current_t)

mesh.points = deform(current_t, initial_disease)
mesh["disease"] = initial_disease
mesh["tissue_rgb"] = make_tissue_colors(initial_disease, current_t)

# Log initial state to audit trail
maybe_log_stage(current_t)


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
    specular=0.4,
    specular_power=25,
    ambient=0.15,
    diffuse=0.85,
)

plotter.add_axes()
plotter.enable_eye_dome_lighting()
plotter.enable_parallel_projection()

plotter.reset_camera()
plotter.camera_position = [
    (0.0, -1.6, 0.7),
    tuple(center),
    (0, 0, 1),
]

plotter.camera.zoom(1.55)

plotter.add_text(
    get_stage_label(current_t),
    position="upper_left",
    font_size=16,
    color="black",
    name="stage_label",
)

plotter.add_text(
    build_explanation_panel(current_t),
    position="lower_left",
    font_size=9,
    color="black",
    name="explanation_label",
)

plotter.add_text(
    get_mapping_text(),
    position="upper_right",
    font_size=10,
    color="black",
    name="mapping_label",
)


# ============================================================
# UPDATE CALLBACK
# ============================================================

def update(val):
    global current_t

    current_t = float(val)

    disease = compute_disease(current_t)
    new_points = deform(current_t, disease)
    new_colors = make_tissue_colors(disease, current_t)

    mesh.points = new_points
    mesh["disease"] = disease
    mesh["tissue_rgb"] = new_colors

    plotter.add_text(
        get_stage_label(current_t),
        position="upper_left",
        font_size=16,
        color="black",
        name="stage_label",
    )

    plotter.add_text(
        build_explanation_panel(current_t),
        position="lower_left",
        font_size=9,
        color="black",
        name="explanation_label",
    )

    maybe_log_stage(current_t)
    plotter.render()


# ============================================================
# SLIDER
# ============================================================

plotter.add_slider_widget(
    update,
    [0.0, 1.0],
    value=current_t,
    title="Disease Progression",
    pointa=(0.35, 0.90),
    pointb=(0.85, 0.90),
)

plotter.show()

# Save audit log if flag was passed
if args.save_audit:
    audit.save_audit_log()
