from sqlalchemy import create_engine, func, JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
import os
from datetime import datetime

DATABASE_URL = os.getenv("DATABASE_URL",  "sqlite:///./test.db") #το δευτερο για να το τρεχω τοπικα εκτος docker
#DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://") # το railway θελει postgresql:// ενω το psycopg2 θελει postgresql+psycopg2://

engine = create_engine(DATABASE_URL, echo=True)

Session = sessionmaker(engine)

class Base(DeclarativeBase):
    pass

def get_db():
    db = Session()
    try:
        yield db
    finally:
        db.close()

class CustomerPred(Base):
    __tablename__ = "customer_predictions_table"

    id: Mapped[int] = mapped_column(primary_key = True)
    churn: Mapped[int] = mapped_column()
    probability: Mapped[float] = mapped_column()
    timestamp: Mapped[datetime] = mapped_column(server_default=func.now())
    input_data: Mapped[dict] = mapped_column(JSON)