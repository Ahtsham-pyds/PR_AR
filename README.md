# Simple FastAPI + SQLite Form App

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Use

1. Open `http://127.0.0.1:8000/`
2. Fill the form and submit
3. Data is stored in SQLite database file `app.db`

## API

- `POST /proposals`
  - Body fields: `id`, `name`, `category`, `description`, `date`, `owner`
