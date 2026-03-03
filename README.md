# DLWeek

Base project for an AI-powered student learning assistant.

## Project Structure

```text
dlweek/
├─ README.md
└─ frontend/
	├─ src/
	├─ package.json
	├─ index.html
	├─ vite.config.ts
	└─ tsconfig*.json
```

## Frontend (React + TypeScript + Vite)

### 1) Go to frontend folder

```bash
cd frontend
```

### 2) Install dependencies

```bash
npm install
```

### 3) Run development server

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r "requirements.txt"
```

3. Add `.env` files

- Add to `performance_agent/` and `converter_agent/`:

```env
OPENAI_API_KEY=<INSERT_YOUR_OWN_OPENAI_API_KEY>
COSMOS_ENDPOINT=https://forfunners.documents.azure.com:443/
COSMOS_KEY=<INSERT_YOUR_OWN_DB_KEY>
COSMOS_DB_NAME=learning_db
COSMOS_OBJECTIVES_CONTAINER=objective
COSMOS_STUDENTS_CONTAINER=student
```

For DLWeek testers, `COSMOS_KEY` is located in Section 4 of the project documentation. You are still required to use your own OpenAI API Key.

4. Run the backend server

```bash
npm run build
```

### 5) Preview production build

```bash
npm run preview
```

## Current UI Base

The starter frontend currently includes:

- Sidebar navigation
- Dashboard hero section
- Key insight metric cards
- Weak concept list
- Actionable recommendation list
- Recent learning activity panel

This provides a clean base to connect real student-learning data and AI-generated recommendations next.
