import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from db import engine

FEATURES = ["worker_id", "task_id", "experience_months"]


def _make_model():
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), ["worker_id", "task_id"])],
        remainder="passthrough",
    )
    return Pipeline([("pre", pre), ("reg", RandomForestRegressor(n_estimators=200, random_state=42))])


def train_and_save():
    """DB의 모든 실적으로 모델을 다시 학습해서 model.joblib에 저장"""
    sql = """
    SELECT r.worker_id, r.task_id, w.experience_months,
           r.hours, r.actual_qty, r.error_rate
    FROM records r
    JOIN workers w ON r.worker_id = w.worker_id
    """
    df = pd.read_sql(sql, engine)
    df["qty_per_hour"] = df["actual_qty"] / df["hours"]
    X = df[FEATURES]

    models, mae = {}, {}
    for target in ["qty_per_hour", "error_rate"]:
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        check = _make_model().fit(X_train, y_train)
        mae[target] = round(float(mean_absolute_error(y_test, check.predict(X_test))), 2)
        models[target] = _make_model().fit(X, y)   # 최종본은 전체 데이터로 학습