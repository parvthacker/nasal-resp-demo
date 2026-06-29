# Thermal Nasal Respiration Analysis

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/parvthacker/nasal-resp-demo/blob/main/NasalRespiration.ipynb)

Contactless respiratory signal extraction from thermal infrared video using YOLO-based nostril localisation. Per-frame mean pixel intensity is sampled from the detected nostril ROI to reconstruct a breathing waveform — no sensors, no contact.

> ⚠️ The notebook is an **educational demos, not medical devices.** 
> Do not use it for diagnosis or any clinical decision.

---

## How It Works

```
Thermal AVI → YOLO detection → Nostril ROI crop → Mean intensity → Respiration signal
```

Each frame is passed through a YOLO detector that finds the face and nostril bounding boxes. The mean thermal intensity of the nostril ROI is recorded per frame, producing a raw respiration waveform at the video's native frame rate (25 Hz). A containment metric checks that the nostril box sits correctly within the face box as a data quality flag.

**Outputs per video:**
- `nasal-resp/*.npy` — raw respiration signal (mean intensity per frame)
- `nostril-visible/*.npy` — binary visibility flag per frame
- `nasal-respf_sig/*_signal.csv` — signal with time axis in seconds
- `annotated_videos_sig/*_annotated.avi` — video with live waveform overlay

---

## Repo Structure

```
├── NasalRespiration.ipynb
├── models/
│   ├── best.pt               # primary YOLO detector
│   ├── NoseLoc_200.pt
│   └── 7Class_AllMS.pt
└── P01_1/
    └── thermal_rec.avi
```

---

## Running in Colab

Click the **Open in Colab** badge above.
Then **Runtime → Run all**.

> Colab's free tier includes a T4 GPU. Make sure to select it under **Runtime → Change runtime type → T4 GPU** before running — the model runs on CUDA.

---

## Running Locally

```bash
git clone https://github.com/parvthacker/nasal-resp-demo.git
cd nasal-resp-demo
pip install ultralytics opencv-python scipy numpy pandas tqdm matplotlib
jupyter notebook NasalRespiration.ipynb
```

---

## Configuration

At the top of the notebook, set:

```python
model = YOLO('./models/best.pt', task='detect').to('cuda')
data_path = "."
pids = ["P01_1"]   # list of participant folder names
```

To add participants, place each folder under `data_path` with the same structure as `P01_1/` and add the folder name to `pids`.

---

## Pipeline Steps

| Notebook Section | What it does |
|---|---|
| **1. Configuration** | Load model, set paths and participant IDs |
| **2. Nasal Localizer** | Runs YOLO on every frame, extracts nostril visibility and containment metrics, saves annotated video with visibility timeline strip |
| **2.1 Localisation Plot** | Per-participant nostril visibility summary, flags participants with >50% invisible frames |
| **3. Resp Signal Extraction** | Re-runs detection and saves mean nostril intensity as the respiration signal alongside a dual-panel annotated video (waveform + visibility) |

---

## Output Video Format

Each annotated video has the original thermal frame on top and a dark strip below with two live panels:

- **Cyan — Thermal Respiration Signal:** rolling normalised waveform of mean nostril intensity. Gaps appear where the nostril is not detected.
- **Green — Nostril Visibility:** binary timeline, high when nostril is detected, low otherwise.

---

## Dependencies

| Package | Purpose |
|---|---|
| `ultralytics` | YOLO inference |
| `opencv-python` | Video I/O and annotation |
| `scipy` | Butterworth filters |
| `numpy` | Array processing |
| `pandas` | Summary dataframes |
| `tqdm` | Progress bars |
| `matplotlib` | Signal plots |
