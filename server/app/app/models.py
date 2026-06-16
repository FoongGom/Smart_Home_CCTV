from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Event(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    event_type = db.Column(db.String(20))  # 'normal', 'anomaly'
    confidence = db.Column(db.Float)
    motion_score = db.Column(db.Float)
    image_path = db.Column(db.String(200))
    description = db.Column(db.Text)

    def __repr__(self):
        return f'<Event {self.timestamp} - {self.event_type}>'
