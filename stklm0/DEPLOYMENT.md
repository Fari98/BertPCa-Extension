# BertPCa STKLM0 App — Deployment Guide

## Prerequisites

All commands run from the **repo root** unless noted otherwise.

Install dependencies (once):
```bash
pip install streamlit
# or, from inside bertpca/:
uv sync --extra research
pip install streamlit
```

---

## 1. Run locally (single machine)

```bash
python run_stklm0_app.py
```

Opens at `http://localhost:8501`.

---

## 2. Share on a local network (hospital / lab LAN)

Run the app so it listens on all network interfaces:

```bash
streamlit run stklm0/app.py --server.address 0.0.0.0 --server.port 8501
```

Anyone on the same network can then open `http://<your-ip>:8501`.

Find your IP on Windows:
```
ipconfig   →  look for "IPv4 Address"
```

> **Note:** This is not encrypted. Do not expose patient data this way outside a trusted network.

---

## 3. Deploy to Streamlit Community Cloud (free, public URL)

This is the easiest option if your code is on GitHub.

### Steps

1. **Push the repo to GitHub** (make sure `stklm0/app.py` is committed).

2. **Upload model files** to the repo or to a cloud storage bucket.
   The models are large (`.keras` files). Options:
   - Git LFS: `git lfs track "*.keras"` then commit as normal.
   - Or store them in `stklm0/outputs/models/` and add to `.gitignore` if too large,
     then download them at startup (see note below).

3. **Go to** [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.

4. Click **New app** → select your repo → set:
   - **Main file path**: `stklm0/app.py`
   - **Python version**: 3.10 or 3.11

5. Add a `stklm0/requirements.txt` (Streamlit Cloud reads this automatically):
   ```
   streamlit>=1.35.0
   pandas>=2.0.0
   numpy>=1.24.0
   tensorflow>=2.13.0
   scikit-learn>=1.3.0
   lifelines>=0.27.0
   datasets>=2.14.0
   pyyaml>=6.0
   ```

6. Click **Deploy**. The app gets a public URL like `https://yourname-bertpca.streamlit.app`.

> **Model files too large for GitHub?** Add a `stklm0/startup.py` that downloads them
> from Google Drive / S3 / Hugging Face Hub on first run, and call it at the top of `app.py`.

---

## 4. Deploy with Docker (self-hosted, any server)

### Dockerfile

Create `stklm0/Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir \
    streamlit>=1.35.0 \
    pandas>=2.0.0 \
    numpy>=1.24.0 \
    tensorflow>=2.13.0 \
    scikit-learn>=1.3.0 \
    lifelines>=0.27.0 \
    datasets>=2.14.0 \
    pyyaml>=6.0

EXPOSE 8501

CMD ["streamlit", "run", "stklm0/app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
```

### Build and run

```bash
# From repo root:
docker build -t bertpca-stklm0 -f stklm0/Dockerfile .
docker run -p 8501:8501 bertpca-stklm0
```

App is then at `http://localhost:8501` (or your server's IP).

### Persist outputs across restarts

```bash
docker run -p 8501:8501 \
  -v $(pwd)/stklm0/outputs:/app/stklm0/outputs \
  bertpca-stklm0
```

This mounts the outputs folder so saved models and predictions survive container restarts.

---

## 5. Where results are saved

| Output | Path |
|--------|------|
| Inference predictions (CSV) | `stklm0/outputs/predictions/predictions_{outcome}_{timestamp}.csv` |
| C-index table (CSV) | `stklm0/outputs/results/c_index_stklm0_train_{timestamp}.csv` |
| Trained model (Keras) | `stklm0/outputs/models/app_trained_stklm0_csm.keras` |
| Milan models (pre-trained) | `stklm0/outputs/models/best_model_milan_{bcr,csm}.keras` |

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `Model not found` | Run `python stklm0/scripts/train_milan.py` first |
| `Missing preprocessing_params.json` | Run `python stklm0/scripts/prepare_stklm0.py --input data/stklm0.csv` |
| `ModuleNotFoundError: bertpca` | Run from the repo root, not from inside `stklm0/` |
| Training OOM | Reduce `batch_size` in `stklm0/config/config_stklm0.yaml` |
