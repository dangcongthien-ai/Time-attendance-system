from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text, func

from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    face_dir = Column(String, nullable=True)
    role = Column(String, nullable=False, default="employee")
    created_at = Column(DateTime, server_default=func.now())


class AttendanceRow(Base):
    __tablename__ = "attendance"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=True)
    name = Column(String, nullable=True)
    action = Column(String, nullable=True)
    ts = Column(DateTime, server_default=func.now())
    raw_time = Column(String, nullable=True)


class Setting(Base):
    __tablename__ = "settings"

    id = Column(Integer, primary_key=True)
    shift_start = Column(String, default="09:00")
    shift_end = Column(String, default="17:00")
    min_in_to_out = Column(Integer, default=60)
    min_out_to_in = Column(Integer, default=60)
    overlay_duration = Column(Integer, default=5)
    camera_source = Column(String, default="0")
    admin_password = Column(String, default="admin")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    actor = Column(String, nullable=True)
    action = Column(String, nullable=False)
    entity = Column(String, nullable=True)
    entity_id = Column(String, nullable=True)
    detail = Column(Text, nullable=True)
    success = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, server_default=func.now())
