import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)  # 42로 고정하면 실행할 때마다 같은 데이터가 나옵니다

# 1) 작업자 6명 (경력이 길수록 빠르고 실수가 적다고 가정)
workers = pd.DataFrame({
    "worker_id": [1, 2, 3, 4, 5, 6],
    "name": ["김민수", "이영희", "박철수", "최지은", "정하늘", "한도윤"],
    "experience_months": [24, 3, 12, 36, 6, 18],
})

# 2) 작업 3종류 (시간당 기본 처리량, 기본 불량률)
tasks = pd.DataFrame({
    "task_id": [1, 2, 3],
    "task_name": ["포장", "조립", "검수"],
    "base_per_hour": [40, 20, 30],
    "base_error_rate": [0.02, 0.04, 0.01],
})

# 3) 사람마다 작업 궁합이 다르다고 가정 (이 값을 AI가 나중에 알아맞히게 됩니다)
affinity = rng.uniform(0.8, 1.2, size=(len(workers), len(tasks)))

# 4) 최근 60 평일 동안의 실적 기록 만들기
days = pd.bdate_range(end="2026-11-16", periods=60)
rows = []
for day in days:
    for wi, w in workers.iterrows():
        ti = rng.integers(0, len(tasks))            # 오늘 맡은 작업 (랜덤)
        t = tasks.iloc[ti]
        hours = int(rng.choice([4, 6, 8]))          # 근무시간

        exp = min(w["experience_months"], 36)
        speed = 0.7 + exp / 36 * 0.5                # 경력이 길수록 빠름
        err_factor = 1.6 - exp / 36 * 0.9           # 경력이 길수록 실수 적음
        fit = affinity[wi, ti]

        target = int(t["base_per_hour"] * hours)    # 목표 작업량
        expected = t["base_per_hour"] * hours * speed * fit
        actual = max(0, int(rng.normal(expected, expected * 0.05)))  # 실제 작업량

        err_rate = min(1.0, t["base_error_rate"] * err_factor / fit)
        defects = int(rng.binomial(actual, err_rate))  # 불량 개수

        rows.append({
            "work_date": day.date(),
            "worker_id": w["worker_id"],
            "task_id": t["task_id"],
            "hours": hours,
            "target_qty": target,
            "actual_qty": actual,
            "defect_qty": defects,
        })

records = pd.DataFrame(rows)
records["achievement_rate"] = (records["actual_qty"] / records["target_qty"] * 100).round(1)
records["error_rate"] = (records["defect_qty"] / records["actual_qty"] * 100).round(2)

# 5) 파일로 저장
os.makedirs("data", exist_ok=True)
workers.to_csv("data/workers.csv", index=False, encoding="utf-8-sig")
tasks.to_csv("data/tasks.csv", index=False, encoding="utf-8-sig")
records.to_csv("data/records.csv", index=False, encoding="utf-8-sig")

print("저장 완료:", len(records), "건")
print()
# 작업자별 평균 달성률/오류율 확인
summary = records.merge(workers, on="worker_id").groupby("name")[["achievement_rate", "error_rate"]].mean().round(1)
print(summary)