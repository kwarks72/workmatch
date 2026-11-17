from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
from sqlalchemy import text

from allocator import allocate, reload_models
from db import engine
from trainer import train_and_save

app = FastAPI(title="WorkMatch API")


@app.get("/workers")
def get_workers():
    """작업자 명단"""
    df = pd.read_sql("SELECT * FROM workers", engine)
    return df.to_dict(orient="records")


@app.get("/records")
def get_records(limit: int = 20):
    """최근 실적 기록 (limit개)"""
    sql = text("SELECT * FROM records ORDER BY work_date DESC LIMIT :n")
    df = pd.read_sql(sql, engine, params={"n": limit})
    return df.to_dict(orient="records")

@app.get("/summary/workers")
def worker_summary():
    """작업자별 평균 달성률 / 오류율"""
    sql = """
    SELECT w.name,
           COUNT(*) AS work_days,
           ROUND(CAST(AVG(r.achievement_rate) AS numeric), 1) AS avg_achievement,
           ROUND(CAST(AVG(r.error_rate) AS numeric), 2) AS avg_error
    FROM records r
    JOIN workers w ON r.worker_id = w.worker_id
    GROUP BY w.name
    ORDER BY avg_achievement DESC
    """
    return pd.read_sql(sql, engine).to_dict(orient="records")


@app.get("/summary/worker-task")
def worker_task_summary():
    """작업자 x 작업종류별 시간당 처리량 / 오류율 (= 개인 능력치)"""
    sql = """
    SELECT w.name, t.task_name,
           COUNT(*) AS times,
           ROUND(CAST(AVG(r.actual_qty * 1.0 / r.hours) AS numeric), 1) AS qty_per_hour,
           ROUND(CAST(AVG(r.error_rate) AS numeric), 2) AS avg_error
    FROM records r
    JOIN workers w ON r.worker_id = w.worker_id
    JOIN tasks t ON r.task_id = t.task_id
    GROUP BY w.name, t.task_name
    ORDER BY w.name, t.task_name
    """
    return pd.read_sql(sql, engine).to_dict(orient="records")





class AllocateRequest(BaseModel):
    avail_hours: dict[str, float]
    demand: dict[str, float]


@app.post("/allocate")
def post_allocate(req: AllocateRequest):
    """근무시간 + 주문 물량 -> 작업자별 자동 배분"""
    return allocate(req.avail_hours, req.demand)


class RecordIn(BaseModel):
    work_date: str
    name: str
    task_name: str
    hours: float
    actual_qty: int
    defect_qty: int


@app.post("/records")
def add_record(rec: RecordIn):
    """오늘 실적 1건 저장"""
    if rec.hours <= 0 or rec.actual_qty <= 0 or rec.defect_qty < 0:
        raise HTTPException(status_code=400, detail="근무시간·실제 수량은 1 이상, 불량 수는 0 이상이어야 합니다")
    with engine.begin() as conn:
        w = conn.execute(text("SELECT worker_id FROM workers WHERE name = :n"), {"n": rec.name}).fetchone()
        t = conn.execute(text("SELECT task_id, base_per_hour FROM tasks WHERE task_name = :n"),
                         {"n": rec.task_name}).fetchone()
        if w is None or t is None:
            raise HTTPException(status_code=404, detail="작업자 또는 작업 이름을 찾을 수 없습니다")
        target = int(t.base_per_hour * rec.hours)
        achievement = round(rec.actual_qty / target * 100, 1)
        error_rate = round(rec.defect_qty / rec.actual_qty * 100, 2)
        conn.execute(text("""
            INSERT INTO records (work_date, worker_id, task_id, hours, target_qty,
                                 actual_qty, defect_qty, achievement_rate, error_rate)
            VALUES (:d, :w, :t, :h, :tg, :a, :df, :ar, :er)
        """), {"d": rec.work_date, "w": w.worker_id, "t": t.task_id, "h": rec.hours,
               "tg": target, "a": rec.actual_qty, "df": rec.defect_qty,
               "ar": achievement, "er": error_rate})
    return {"saved": True, "achievement_rate": achievement, "error_rate": error_rate}


@app.post("/retrain")
def retrain():
    """쌓인 실적으로 AI 다시 학습"""
    result = train_and_save()
    reload_models()
    return result


@app.get("/dashboard")
def dashboard():
    """대시보드 화면"""
    return FileResponse("static/dashboard.html")