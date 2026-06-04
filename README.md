# 🫁 Organ Disease Progression Visualiser
> **CT-Driven 3D Lung Disease Progression — COMP8851 | Macquarie University 2026**

---

*Can you tell what stage this disease is at just from a CT scan?*
*Most people can't. This system can — and it shows you exactly why.*

---

## What This Is

A deterministic pipeline that processes real chest CT images and maps three 
quantitative signals directly onto an anatomical 3D lung mesh.

| What goes in | What comes out |
|---|---|
| Any chest CT image (JPG or PNG) | Live 3D lung model at matched disease stage |
| Severity — mean pixel intensity | Colour strength + structural deformation |
| Texture — std deviation of pixels | Surface roughness of the mesh |
| HDR — proportion of pixels > 0.6 | Lesion spread across the lung surface |

**Score = 0.65 × Severity + 0.20 × Texture + 0.15 × HDR**

Range: `0.218` (near healthy) → `0.809` (advanced fibrosis)  
Every number on screen came from a pixel in the CT image.  
Every decision is logged. Every transformation is explained.

---

## See It In Action

| Case 08 — Early Disease (score: 0.218) | Case 01 — Advanced Disease (score: 0.809) |
|---|---|
| <img width="620" alt="Case 08" src="https://github.com/user-attachments/assets/22d362a1-840b-4941-9d7f-c360ba42639e" /> | <img width="623" alt="Case 01" src="https://github.com/user-attachments/assets/c14b4e3d-86cb-49c5-b4d0-b7ced4ea724c" /> |
| Near-healthy. Light pink. Smooth surface. Minimal deformation. | Deep fibrotic red. Volume loss. Strong deformation. HDR dominant. |

---

## All 8 Cases

<img width="1512" alt="Case 1" src="https://github.com/user-attachments/assets/1ab91dd7-7a6c-4aac-b4a9-e21623a3c13f" />
<img width="1512" alt="Case 2" src="https://github.com/user-attachments/assets/30451161-aae5-44c1-b9d7-4be1071b4ecb" />
<img width="1512" alt="Case 3" src="https://github.com/user-attachments/assets/5075b20d-24d1-41b9-b2b7-3442af19e7ff" />
<img width="1512" alt="Case 4" src="https://github.com/user-attachments/assets/19d05a5a-0403-439a-8dfc-d8940b3eef2e" />
<img width="1512" alt="Case 5" src="https://github.com/user-attachments/assets/725163e9-3e53-4fda-802e-e54cc5c0416a" />
<img width="1512" alt="Case 6" src="https://github.com/user-attachments/assets/5579d3f6-b1c5-4c05-b495-a4bc267f0c00" />

---

## Run It

```bash
# Drop any CT image in and run
python "Integrated Pipeline.py" case_01_image.jpg

# See the segmentation working live
python "Integrated Pipeline.py" case_01_image.jpg --show_ct

# Save the audit trail on exit
python "Integrated Pipeline.py" case_01_image.jpg --save_audit
```

> `lung_model.glb` must be in the same folder. PNG cases use `.png` extension.

---

## How It Works

```
CT Image  ──▶  Segment Lungs  ──▶  Extract 3 Signals  ──▶  Score  ──▶  3D Mesh  ──▶  Explain  ──▶  Audit Log
```

The pipeline extracts three numbers from each CT image:

| Signal | What It Measures | Why It Matters |
|--------|-----------------|----------------|
| **Severity** | Mean pixel intensity | Overall tissue density and abnormality |
| **Texture** | Standard deviation | Structural irregularity and heterogeneity |
| **HDR** | Proportion of pixels > 0.6 | Dense lesion and fibrosis coverage |

These combine into a weighted progression score:

```
Score = 0.65 × Severity + 0.20 × Texture + 0.15 × HDR
```

That score drives everything — colour, lesion spread, deformation, roughness.

---

## Results Across All 8 Cases

| Case | Severity | Texture | HDR | Score | Stage |
|------|----------|---------|-----|-------|-------|
| Case 01 | 0.4201 | 0.1568 | 0.1168 | **0.809** | 🔴 Advanced |
| Case 02 | 0.3537 | 0.1556 | 0.0766 | **0.668** | 🟠 Moderate |
| Case 03 | 0.3255 | 0.0985 | 0.0218 | **0.409** | 🟡 Early |
| Case 04 | 0.4603 | 0.0932 | 0.0774 | **0.754** | 🔴 Advanced |
| Case 05 | 0.2713 | 0.0745 | 0.0097 | **0.242** | 🟡 Early |
| Case 06 | 0.3070 | 0.1209 | 0.0324 | **0.436** | 🟡 Early |
| Case 07 | 0.2772 | 0.1357 | 0.0411 | **0.423** | 🟡 Early |
| Case 08 | 0.1970 | 0.1242 | 0.0219 | **0.218** | 🟡 Early |

---

## Clinical Accountability — The Audit Trail

Every run with `--save_audit` produces a JSON file that records the full pipeline trail:

```json
{
  "pipeline_stage": "1_CT_EXTRACTION",
  "event_data": {
    "severity": 0.4201,
    "texture": 0.1568,
    "hdr": 0.1168,
    "lung_pixel_count": 7937
  }
},
{
  "pipeline_stage": "2_FEATURE_MAPPING",
  "event_data": {
    "progression_score": 0.809,
    "weighting": "0.65*severity + 0.20*texture + 0.15*hdr"
  }
},
{
  "pipeline_stage": "3_STAGE_TRANSITION",
  "event_data": {
    "disease_stage": "Stage: Advanced Disease",
    "lesion_radius_multiplier": 6.152,
    "deformation_multiplier": 0.885
  }
},
{
  "pipeline_stage": "4_VERTEX_MODIFICATION",
  "event_data": {
    "affected_vertex_count": 264370,
    "clinical_justification": "Vertex shifts represent fibrotic contraction, pleural surface indentation, and basal collapse — consistent with IPF morphology."
  }
}
```

**8 cases. 8 audit logs. Every decision traceable.**

---

## Development Journey

This system was built incrementally — each script solving one problem before the next:

| Script | What It Solved | When |
|--------|---------------|------|
| `ct_step3_lung_mask.py` | Getting data out of CT images | Preliminary |
| `prototype.py` | Building the 3D simulation | Preliminary |
| `ct_driven_visualizer.py` | Connecting CT data to 3D model | Post-preliminary |
| `Keyboard_switching.py` | Switching between all 8 cases live | Post-preliminary |
| `Integrated Pipeline.py` | One command, live extraction, full audit | **Final** |

---

## Project Structure

```
Disease Progression Visualization/
│
├── Integrated Pipeline.py       ← Run this
├── ct_driven_visualizer.py
├── Keyboard_switching.py
├── prototype.py
├── ct_step3_lung_mask.py
├── progression_engine.py
│
├── lung_model.glb               ← Human Atlas reference mesh
│
├── case_01_image.jpg  ── case_08_image.jpg
│
└── audit_logs/
    └── audit_log_case_XX_image_xxxx.json  (×8)
```

---

## Tech Stack

`Python` `OpenCV` `NumPy` `PyVista` `Trimesh` `Matplotlib`

---

## Team

**Supervisor:** Filippo Cenacchi — filippo.cenacchi@mq.edu.au

| Member | Role |
|--------|------|
| Mahesh Sai Kandula | Technical Lead — Full system implementation |
| David Kong | Research Lead — Literature, audit logger design |
| Chirag Srinivasamurthy | AI/NLP — ClinicalBERT text extraction |
| Damaranath Kokkula | Research — Domain research, Blender exploration |
| Anil Kumar Varada | Analysis — Feature-to-mesh mapping concepts |

**GitHub:** https://github.com/MSkandula/Disease-Progression-visuliazation.git

---

*COMP8851 S1 2026 — Macquarie University*
