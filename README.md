# FAILSAFE — Early Student Failure Risk Detection System

**Prem Kadam · Bhavit · Krishna Jain | IIT Guwahati**

XGBoost + SHAP explainability. Predicts at-risk students from academic and behavioural data. FastAPI backend, single-file React frontend.

---

## Running locally

### 1. Place the model
Copy `failsafe_model.pkl` (exported from the training notebook) into this folder.

```
failsafe/
├── main.py
├── index.html
├── requirements.txt
├── render.yaml
├── failsafe_model.pkl   ← put it here
└── README.md
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Start the server
```bash
uvicorn main:app --reload
```

### 4. Open the dashboard
Go to **http://localhost:8000** in your browser.

Upload `student-mat.csv` or `student-por.csv`. The dashboard loads instantly.

---

## Deploying for demo / submission

No one should have to run terminal commands just to see the dashboard. Deploy it in two steps.

### Step 1 — Backend on Render (free)

Render will host the FastAPI server and the trained model.

1. Push this entire folder to a GitHub repository (include `failsafe_model.pkl`).
2. Go to [render.com](https://render.com) and sign up with GitHub.
3. Click **New → Web Service** and connect your repository.
4. Render auto-detects `render.yaml` — no configuration needed.
5. Click **Deploy**. Wait ~2 minutes.
6. Copy your service URL — it will look like `https://failsafe-api.onrender.com`.

> **Note:** The free tier sleeps after 15 minutes of inactivity. First request after sleep takes ~30 seconds to wake up. Subsequent requests are instant. Fine for a demo.

### Step 2 — Frontend on Netlify (free)

1. Open `index.html` in a text editor.
2. Find this line near the top of the script (line ~34):
   ```javascript
   const API_BASE = '';
   ```
3. Replace the empty string with your Render URL:
   ```javascript
   const API_BASE = 'https://failsafe-api.onrender.com';
   ```
4. Save the file.
5. Go to [netlify.com](https://netlify.com), sign up, and drag `index.html` into the deploy box.
6. Netlify gives you a public URL in under a minute — e.g. `https://failsafe-prem.netlify.app`.

Share that Netlify URL with verifiers. They just open the link, upload a CSV, and see the dashboard. No setup required on their end.

---

## API reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/`              | Redirects to dashboard |
| `GET`  | `/app`           | Serves the React frontend |
| `POST` | `/upload`        | Upload CSV → predictions |
| `GET`  | `/students`      | All students, sorted by risk |
| `GET`  | `/students/{id}` | One student — SHAP + interventions |
| `GET`  | `/summary`       | Aggregate stats |
| `GET`  | `/docs`          | Auto-generated API docs |

---

## Tech stack

| Layer | Technology |
|-------|------------|
| ML model | XGBoost + Optuna (trained in Colab) |
| Explainability | SHAP TreeExplainer |
| Backend | FastAPI + SQLite |
| Frontend | React 18 + Tailwind CSS (CDN, no build step) |
| Backend hosting | Render |
| Frontend hosting | Netlify |
