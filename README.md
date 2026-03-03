# Virtuoso

Virtuoso is a all-in-one platform to optimize learning. Built for students who wants to reach their skill objectives, Virtuoso uses a user-friendly user interface (UI) along with an AI-powered pipeline to produce a recommender for students to maximize their learning.

## Quick Setup

1. Clone this repository to a folder of your choice

```bash
git clone https://github.com/lemonishi/dlweek
```

2. Create a Python environment

- For Anaconda users, create a new Conda environment:

```bash
conda create --name dlweek python=3.10
conda activate dlweek
```

- Otherwise, create a virtual environment

```bash
python -m venv .venv
source venv/bin/activate
```

3. Install Python dependencies from the project root

```bash
pip install -r requirements.txt
```

4. Add `.env` files

- Add 1 in each of the `performance_agent` and `converter_agent` folders:

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
cd performance_agent
python api.py
```

5. Start the frontend server

```bash
cd frontend
npm install
npm run dev
```

6. Navigate to the link in the output (should be `http:localhost:5173`)

## Current UI Base

The starter frontend currently includes:

- Sidebar navigation
- Dashboard hero section
- Key insight metric cards
- Weak concept list
- Actionable recommendation list
- Recent learning activity panel

This provides a clean base to connect real student-learning data and AI-generated recommendations next.
