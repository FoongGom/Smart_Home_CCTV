from flask import Flask
from flask_cors import CORS
from app.models import db

def create_app():
    app = Flask(__name__)
    CORS(app)
    
    # ==================== MariaDB 연결 ====================
    app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://LSM:password@127.0.0.1:3306/security_db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    # ====================================================
    
    db.init_app(app)
    
    with app.app_context():
        db.create_all()
    
    from app.routes import bp
    app.register_blueprint(bp)
    
    return app
