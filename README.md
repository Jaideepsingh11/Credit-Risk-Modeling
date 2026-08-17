# Credit Risk Assessment — Web App

A Flask web app that scores a loan applicant into risk segment **P1 (best) → P4
(worst)** using an XGBoost model trained on the two case-study bureau datasets.

```
credit_risk_app/
├── data/
│   ├── case_study1.xlsx        # trade-line summary data
│   └── case_study2.xlsx        # bureau delinquency / enquiry data
├── model/
│   ├── train_model.py          # cleaning + feature selection + training script
│   ├── model.pkl                (generated)
│   ├── scaler.pkl                (generated)
│   ├── label_encoder.pkl         (generated)
│   └── feature_info.json         (generated)
├── app/
│   ├── app.py                  # Flask routes
│   ├── wsgi.py                 # production entry point (gunicorn)
│   ├── templates/
│   │   └── index.html
│   └── static/
│       └── style.css
├── requirements.txt
├── Procfile                    # for Render / Heroku-style platforms
├── .gitignore
└── README.md
```

Why it's split this way:
- **`data/`** — raw inputs only. Never imported by the web app directly.
- **`model/`** — anything related to *producing* a model. Running
  `train_model.py` here regenerates the four artifact files the app needs.
  Keeping this separate from `app/` means you can retrain on a totally
  different machine (e.g. a GPU box) and just copy the four output files
  over to `app/`'s host.
- **`app/`** — the actual web server. It only ever *reads* the artifacts in
  `model/`; it never touches `data/` or retrains anything. This is the only
  folder you deploy to a server.

---

## 1. What was fixed in your original notebook

| Bug | Symptom | Fix |
|---|---|---|
| VIF loop used a fixed `range(0, total_columns)` while columns were being dropped inside the loop | `IndexError` partway through, or (on newer pandas) a `TypeError` inside statsmodels | Rewritten as a `while` loop driven by the *current* number of remaining columns |
| Numeric-column detection used `df[i].dtype != 'object'` | On pandas 2.x/3.x, text columns can get dtype `"str"` instead of `"object"`, so they silently leaked into the "numeric" list and crashed VIF/statsmodels with `ufunc 'isfinite' not supported` | Switched to `df.select_dtypes(include=np.number)`, which is correct regardless of pandas version |
| `df1 = df1.loc[...]` then further in-place edits | `SettingWithCopyWarning` | Added `.copy()` after every boolean-mask filter |
| Chained `df.loc[df['EDUCATION']==...] = n` assignments | `SettingWithCopyWarning`, fragile if a category is spelled differently | Replaced with one `.map(EDUCATION_MAP)` call |
| No fixed random seed / stratification on some splits | Accuracy shifted a little every run | `train_test_split(..., random_state=42, stratify=y_enc)` everywhere |
| Nothing was saved after training | You'd have to retrain every time you wanted to predict on one applicant | `train_model.py` now saves `model.pkl`, `scaler.pkl`, `label_encoder.pkl`, and a `feature_info.json` describing exactly what the model expects |

Re-run end to end, the fixed pipeline reproduces your original notebook's
result almost exactly: **~78% test accuracy**, same per-class precision/recall
pattern (P3 is the hardest class — this is a property of the data, not a bug).

---

## 2. Running it locally

```bash
cd credit_risk_app
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Train the model (writes model/model.pkl etc.)
cd model
python train_model.py
cd ..

# 2. Run the web app
cd app
python app.py
```

Open **http://127.0.0.1:5000** — fill in the applicant's bureau figures and
click **Score applicant**.

There's also a JSON API for programmatic use:
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"NETMONTHLYINCOME": 45000, "AGE": 34, "MARITALSTATUS": "Married", ...}'
```
Any field you omit defaults to `0` (numeric) or the first valid category.

---

## 3. Putting the code in separate folders (already done above)

If you're starting from your original single notebook, here's the mental
model for splitting it up:

1. **Everything before `model.fit(...)`** → goes in `model/train_model.py`,
   wrapped in functions (`load_data`, `clean_data`, `select_numeric_features`,
   `train`) instead of top-to-bottom notebook cells.
2. **The trained objects** (`model`, `scaler`, `label_encoder`) → don't keep
   these in memory between "sessions". `joblib.dump(...)` them to disk once,
   and have the web app `joblib.load(...)` them at startup.
3. **Anything that depends on request input** (a specific applicant's
   numbers) → goes in `app/app.py`. It should *never* re-run training; it
   only loads the saved artifacts and calls `.predict()`.
4. **HTML/CSS** → `app/templates/` and `app/static/` (Flask's default
   convention — `render_template()` looks in `templates/` automatically).

---

## 4. Deploying it

You need two things on the server: the four files inside `model/` (already
trained — you don't need `data/` in production) and everything in `app/`.

### Option A — Render.com (easiest, free tier available)
1. Push this whole `credit_risk_app/` folder to a GitHub repo.
2. On [render.com](https://render.com) → **New → Web Service** → connect the repo.
3. Build command: `pip install -r requirements.txt && python model/train_model.py`
   *(or just commit the already-trained `model/*.pkl` files and skip
   retraining in the build step)*
4. Start command: `gunicorn --chdir app wsgi:app --bind 0.0.0.0:$PORT`
   (this matches the `Procfile` already in the repo, so Render/Heroku detect
   it automatically — you may not even need step 3/4 typed in manually).
5. Deploy. Render gives you a public URL like `https://your-app.onrender.com`.

### Option B — Railway.app
1. Push to GitHub, then **New Project → Deploy from GitHub repo** on
   [railway.app](https://railway.app).
2. Railway auto-detects the `Procfile`. Just make sure `PORT` is read from
   the environment (already handled in the `Procfile`).
3. Add the repo, deploy, done.

### Option C — PythonAnywhere (good for beginners, always-on free tier)
1. Upload the `credit_risk_app/` folder (or `git clone` it) in a Bash console.
2. Create a virtualenv and `pip install -r requirements.txt`.
3. Go to the **Web** tab → **Add a new web app** → **Flask** → point the WSGI
   configuration file to import `app` from `app/app.py`
   (`from app import app as application`).
4. Reload the web app. You get a URL like `yourname.pythonanywhere.com`.

### Option D — Docker (portable, works anywhere: AWS, GCP, DigitalOcean, etc.)
Add this `Dockerfile` at the project root:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY model/ ./model/
COPY app/ ./app/
EXPOSE 8000
CMD ["gunicorn", "--chdir", "app", "wsgi:app", "--bind", "0.0.0.0:8000"]
```
Then:
```bash
docker build -t credit-risk-app .
docker run -p 8000:8000 credit-risk-app
```

### A few production notes
- **Never run with `debug=True`** in production (it's already off in
  `wsgi.py`/gunicorn — that flag is only in `app.py`'s local `__main__`
  block).
- **Retrain offline, deploy the artifacts.** Don't run `train_model.py` on
  every server boot — train once, commit/upload the four files in `model/`,
  and let the web app just load them.
- **Model size**: `model.pkl` here is a small XGBoost model (a few hundred
  KB–few MB), fine to commit directly to git. If it grows large, use
  [Git LFS](https://git-lfs.com/) or store it in S3/GCS and download it on
  first boot instead.
- This model was trained on Indian retail-lending bureau data; if you deploy
  this for real decisioning (rather than as a demo/portfolio project), you'd
  want fairness/bias testing on the protected attributes (GENDER,
  MARITALSTATUS) before using it to influence real credit decisions.
