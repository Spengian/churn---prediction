from sqlalchemy import create_engine, Column, Integer, Float, JSON, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
import os
from datetime import datetime

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./test.db")
DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://")

engine = create_engine(DATABASE_URL)
Base = declarative_base()
Session = sessionmaker(engine)

class CustomerPred(Base):
    __tablename__ = "customer_predictions_table"
    id = Column(Integer, primary_key=True)
    churn_pred = Column(Integer)
    churn_real = Column(Integer, nullable = True)
    probability = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    input_data = Column(JSON)