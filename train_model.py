import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from db import engine

# 1) DB에서 실적 가져오기
sql = """
SELECT r.worker_id, r.task_id, w.experience_months,
       r.hours, r.actual_qty, r.error_rate
FROM records r
JOIN workers w ON r.worker_id = w.worker_id
"""
df = pd.read_sql(sql, engine)
df["qty_per_hour"] = df["actual_qty"] / df["hours"]   # 시간당 처리량

# 2) AI에게 줄 힌트(입력)와 맞혀야 할 정답(출력)
FEATURES = ["worker_id", "task_id", "experience_months"]
X = df[FEATURES]


def make_model(regressor):
    # worker_id, task_id는 숫자가 아니라 '이름표'라서 변환이 필요합니다
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), ["worker_id", "task_id"])],
        remainder="passthrough",
    )
    return Pipeline([("pre", pre), ("reg", regressor)])


models = {}
for target in ["qty_per_hour", "error_rate"]:
    y = df[target]

    # 3) 문제집(80%)으로 공부시키고, 모의고사(20%)로 시험 보기
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 비교 기준: 그냥 평균만 찍는 바보 모델
    baseline = make_model(DummyRegressor(strategy="mean")).fit(X_train, y_train)
    # 진짜 모델: 랜덤 포레스트 (작은 결정 나무 200개의 다수결)
    model = make_model(RandomForestRegressor(n_estimators=200, random_state=42))
    model.fit(X_train, y_train)

    mae_base = mean_absolute_error(y_test, baseline.predict(X_test))
    mae_model = mean_absolute_error(y_test, model.predict(X_test))
    print(f"[{target}] 평균 찍기 오차: {mae_base:.2f}  →  AI 모델 오차: {mae_model:.2f}")

    # 4) 시험이 끝났으니 전체 데이터로 다시 공부시켜서 최종 모델 완성
    model.fit(X, y)
    models[target] = model

# 5) 파일로 저장 (4단계에서 불러다 씀)
joblib.dump(models, "model.joblib")
print("model.joblib 저장 완료")

# 6) 모든 '작업자 x 작업' 조합의 예측 결과표 만들기
workers = pd.read_sql("SELECT worker_id, name, experience_months FROM workers", engine)
tasks = pd.read_sql("SELECT task_id, task_name FROM tasks", engine)
grid = workers.merge(tasks, how="cross")

grid["예측_시간당개수"] = models["qty_per_hour"].predict(grid[FEATURES]).round(1)
grid["예측_오류율"] = models["error_rate"].predict(grid[FEATURES]).round(2)

print()
print("=== 예측 시간당 처리량 ===")
print(grid.pivot(index="name", columns="task_name", values="예측_시간당개수"))
print()
print("=== 예측 오류율(%) ===")
print(grid.pivot(index="name", columns="task_name", values="예측_오류율"))