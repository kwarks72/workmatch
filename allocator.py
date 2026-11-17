import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linprog

from db import engine

FEATURES = ["worker_id", "task_id", "experience_months"]
_models = joblib.load("model.joblib")   # 3단계에서 만든 AI 모델


def reload_models():
    """새로 학습한 모델을 다시 불러오기"""
    global _models
    _models = joblib.load("model.joblib")


def _predict_table():
    """모든 '작업자 x 작업' 조합에 대해 AI가 예측한 처리량/오류율"""
    workers = pd.read_sql("SELECT worker_id, name, experience_months FROM workers", engine)
    tasks = pd.read_sql("SELECT task_id, task_name FROM tasks", engine)
    grid = workers.merge(tasks, how="cross")
    grid["rate"] = _models["qty_per_hour"].predict(grid[FEATURES])
    grid["err"] = _models["error_rate"].predict(grid[FEATURES]).clip(min=0)
    return grid


def allocate(avail_hours: dict, demand: dict):
    """
    avail_hours: 작업자별 오늘 근무가능 시간  예) {"김민수": 8, "최지은": 0}
    demand:      작업별 오늘 필요 물량        예) {"포장": 500, "조립": 200}
    """
    grid = _predict_table()
    names = list(grid["name"].unique())
    task_names = list(grid["task_name"].unique())

    for tn in demand:
        if tn not in task_names:
            raise ValueError(f"알 수 없는 작업: {tn}")

    W, T = len(names), len(task_names)
    rate = np.zeros((W, T))   # 시간당 처리량
    err = np.zeros((W, T))    # 오류율(%)
    for _, r in grid.iterrows():
        w, t = names.index(r["name"]), task_names.index(r["task_name"])
        rate[w, t], err[w, t] = r["rate"], r["err"]

    # 구할 값: h[w,t] = w가 t작업에 쓰는 시간,  s[t] = t작업의 부족 물량
    n = W * T + T
    c = np.zeros(n)
    for w in range(W):
        for t in range(T):
            c[w * T + t] = rate[w, t] * err[w, t] / 100   # 1시간당 예상 불량 수
    c[W * T:] = 1000   # 물량 부족은 불량보다 훨씬 나쁘게 취급 -> 물량부터 채움

    A, b = [], []
    # 제한 1: 각 작업자는 근무가능 시간을 넘길 수 없다
    for w, name in enumerate(names):
        row = np.zeros(n)
        row[w * T:(w + 1) * T] = 1
        A.append(row)
        b.append(avail_hours.get(name, 0))
    # 제한 2: 각 작업의 필요 물량을 채워야 한다 (못 채운 만큼은 s로 표시)
    for t, tn in enumerate(task_names):
        row = np.zeros(n)
        for w in range(W):
            row[w * T + t] = -rate[w, t]
        row[W * T + t] = -1
        A.append(row)
        b.append(-demand.get(tn, 0))

    res = linprog(c, A_ub=A, b_ub=b, bounds=(0, None), method="highs")
    if not res.success:
        raise RuntimeError("배분 계산 실패: " + res.message)

    x = res.x
    schedule = []
    for w, name in enumerate(names):
        for t, tn in enumerate(task_names):
            h = x[w * T + t]
            if h > 0.05:
                qty = h * rate[w, t]
                schedule.append({
                    "name": name,
                    "task_name": tn,
                    "hours": round(float(h), 1),
                    "planned_qty": int(round(qty)),
                    "expected_defects": round(float(qty * err[w, t] / 100), 1),
                })

    summary = []
    for t, tn in enumerate(task_names):
        made = sum(rate[w, t] * x[w * T + t] for w in range(W))
        summary.append({
            "task_name": tn,
            "demand": demand.get(tn, 0),
            "planned_qty": int(round(made)),
            "shortage": int(round(float(x[W * T + t]))),
        })

    total_defects = round(sum(s["expected_defects"] for s in schedule), 1)
    return {"schedule": schedule, "summary": summary, "total_expected_defects": total_defects}


if __name__ == "__main__":
    demand = {"포장": 500, "조립": 200, "검수": 350}
    scenarios = {
        "정상 근무 (전원 8시간)": {
            "김민수": 8, "이영희": 8, "박철수": 8, "최지은": 8, "정하늘": 8, "한도윤": 8,
        },
        "최지은 결근, 김민수 반차": {
            "김민수": 4, "이영희": 8, "박철수": 8, "최지은": 0, "정하늘": 8, "한도윤": 8,
        },
    }
    for title, hours in scenarios.items():
        result = allocate(hours, demand)
        print("=" * 50)
        print(title)
        print(pd.DataFrame(result["schedule"]).to_string(index=False))
        print()
        print(pd.DataFrame(result["summary"]).to_string(index=False))
        print("예상 총 불량:", result["total_expected_defects"], "개")