import pandas as pd
from db import engine

print("저장할 DB:", engine.url.render_as_string(hide_password=True))

for name in ["workers", "tasks", "records"]:
    df = pd.read_csv(f"data/{name}.csv")
    if name == "records":
        df["hours"] = df["hours"].astype(float)   # 0.5시간 단위 입력도 가능하게
    df.to_sql(name, engine, if_exists="replace", index=False)
    print(f"{name} 테이블에 {len(df)}건 저장")