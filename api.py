
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import asyncio
import uuid
import pandas as pd
from pipeline import run_full_pipeline

app = FastAPI()

tasks = {}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...), target_col: str = Form(...)):
    """
    This endpoint accepts a file upload and a target column, adds the analysis tasks to a task queue and returns a task_id.
    """
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "pending", "result": None}
    
    # In a real implementation, this would be a background task
    # For now, we'll run it asynchronously here
    asyncio.create_task(run_analysis(task_id, file, target_col))
    
    return {"task_id": task_id}

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    """
    This endpoint returns the status of a given task.
    """
    task = tasks.get(task_id)
    if not task:
        return JSONResponse(status_code=404, content={"message": "Task not found"})
    return {"status": task["status"]}

@app.get("/results/{task_id}")
async def get_results(task_id: str):
    """
    This endpoint returns the results of a completed analysis in JSON format.
    """
    task = tasks.get(task_id)
    if not task:
        return JSONResponse(status_code=404, content={"message": "Task not found"})
    if task["status"] != "completed":
        return {"status": task["status"], "message": "Analysis is not yet complete."}
    return {"status": task["status"], "result": task["result"]}

async def run_analysis(task_id: str, file: UploadFile, target_col: str):
    """
    This function simulates running the analysis in the background.
    """
    tasks[task_id]["status"] = "in_progress"
    try:
        data = pd.read_csv(file.file)
        report_path = run_full_pipeline(file.filename, data, target_col)
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["result"] = report_path
    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["result"] = str(e)
