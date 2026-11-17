import os

from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()   # .env 파일의 값을 읽어옵니다

# .env에 DATABASE_URL이 있으면 PostgreSQL, 없으면 SQLite를 씁니다
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///workmatch.db")

engine = create_engine(DATABASE_URL)