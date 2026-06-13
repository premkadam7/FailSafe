# FAILSAFE — Early Student Failure Risk Detection System

**Prem Kadam · Bhavit · Krishna Jain | IIT Guwahati**

Predicts which students are at risk of failing before the semester ends, using attendance, grades, and behavioural data. Every prediction is explained using SHAP so faculty know exactly why a student was flagged — and what to do about it.

---

**Live Demo:** https://symphonious-fairy-50ab14.netlify.app/

---

## Running locally

```bash
# 1. Clone the repo and enter the directory
git clone <repo-url>
cd failsafe

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server
uvicorn main:app --reload

# 4. Open in browser
http://localhost:8000
```

Upload `student-mat.csv` or `student-por.csv` from the UCI Student Performance Dataset.

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML model | XGBoost + Optuna (50-trial hyperparameter tuning) |
| Explainability | SHAP TreeExplainer — global + per-student |
| Backend | FastAPI + SQLite |
| Frontend | React 18 + Tailwind CSS |
| Backend hosting | Render |
| Frontend hosting | Netlify |
