from flask import Blueprint, render_template, jsonify, request, send_from_directory
from app.models import db, Event

bp = Blueprint('main', __name__)

@bp.route('/')
def dashboard():
    events = Event.query.order_by(Event.timestamp.desc()).limit(15).all()
    return render_template('dashboard.html', events=events)

@bp.route('/live')
def live():
    return render_template('live.html')

@bp.route('/events')
def events():
    events = Event.query.order_by(Event.timestamp.desc()).limit(100).all()
    return render_template('events.html', events=events)

@bp.route('/static/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('static/images', filename)

@bp.route('/live.jpg')
def live_image():
    return send_from_directory('static', 'live.jpg')

@bp.route('/api/events', methods=['GET'])
def get_events():
    events = Event.query.order_by(Event.timestamp.desc()).limit(100).all()
    return jsonify([{
        'timestamp': e.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        'event_type': e.event_type,
        'confidence': e.confidence,
        'motion_score': e.motion_score,
        'image_path': e.image_path
    } for e in events])

@bp.route('/api/event', methods=['POST'])
def add_event():
    data = request.json
    event = Event(
        event_type=data.get('event_type'),
        confidence=data.get('confidence'),
        motion_score=data.get('motion_score'),
        image_path=data.get('image_path'),
        description=data.get('description', '')
    )
    db.session.add(event)
    db.session.commit()
    return jsonify({'status': 'success'})
