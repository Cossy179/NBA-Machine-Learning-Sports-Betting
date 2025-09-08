#!/usr/bin/env python3
"""
GoonSteen Web Backend
Flask application for NBA sports betting platform
"""

import os
import sqlite3
import hashlib
import secrets
import json
from datetime import datetime, timedelta, timezone
from functools import wraps
import logging
from typing import Optional, Dict, Any

from flask import Flask, request, jsonify, session, g, send_from_directory
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import jwt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='web', static_url_path='')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(32))
app.config['DATABASE'] = os.environ.get('DATABASE_PATH', 'web_database.db')
app.config['JWT_SECRET'] = os.environ.get('JWT_SECRET', secrets.token_hex(32))
app.config['JWT_EXPIRATION_HOURS'] = int(os.environ.get('JWT_EXPIRATION_HOURS', '24'))

# Enable CORS for all routes
CORS(app)

# Database helper functions
def get_db():
    """Get database connection"""
    if 'db' not in g:
        g.db = sqlite3.connect(app.config['DATABASE'])
        g.db.row_factory = sqlite3.Row
    return g.db

def close_db(e=None):
    """Close database connection"""
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db():
    """Initialize database with schema"""
    with app.app_context():
        db = get_db()
        with open('database_schema.sql', 'r') as f:
            db.executescript(f.read())
        db.commit()
        logger.info("Database initialized successfully")

def query_db(query, args=(), one=False):
    """Execute database query"""
    db = get_db()
    cur = db.execute(query, args)
    rv = cur.fetchall()
    cur.close()
    return (rv[0] if rv else None) if one else rv

def execute_db(query, args=()):
    """Execute database command"""
    db = get_db()
    cur = db.execute(query, args)
    db.commit()
    return cur.lastrowid

# Authentication helpers
def generate_session_token():
    """Generate secure session token"""
    return secrets.token_urlsafe(32)

def hash_password(password: str, salt: str = None) -> tuple:
    """Hash password with salt"""
    if salt is None:
        salt = secrets.token_hex(16)
    
    password_hash = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        100000
    )
    return password_hash.hex(), salt

def verify_password(password: str, password_hash: str, salt: str) -> bool:
    """Verify password against hash"""
    computed_hash, _ = hash_password(password, salt)
    return secrets.compare_digest(computed_hash, password_hash)

def create_jwt_token(user_id: int) -> str:
    """Create JWT token for user"""
    payload = {
        'user_id': user_id,
        'exp': datetime.now(timezone.utc) + timedelta(hours=app.config['JWT_EXPIRATION_HOURS']),
        'iat': datetime.now(timezone.utc)
    }
    return jwt.encode(payload, app.config['JWT_SECRET'], algorithm='HS256')

def verify_jwt_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify JWT token and return payload"""
    try:
        payload = jwt.decode(token, app.config['JWT_SECRET'], algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

# Authentication decorators
def login_required(f):
    """Decorator to require authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if token:
            token = token.replace('Bearer ', '')
        else:
            token = session.get('token')
        
        if not token:
            return jsonify({'error': 'Authentication required'}), 401
        
        payload = verify_jwt_token(token)
        if not payload:
            return jsonify({'error': 'Invalid or expired token'}), 401
        
        # Get user from database
        user = query_db('SELECT * FROM users WHERE id = ?', [payload['user_id']], one=True)
        if not user or user['status'] != 'active':
            return jsonify({'error': 'User not found or inactive'}), 401
        
        g.current_user = dict(user)
        return f(*args, **kwargs)
    
    return decorated_function

def admin_required(f):
    """Decorator to require admin privileges"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not g.current_user or not g.current_user.get('is_admin'):
            return jsonify({'error': 'Admin privileges required'}), 403
        return f(*args, **kwargs)
    
    return decorated_function

# Utility functions
def log_user_activity(user_id: int, activity_type: str, description: str, metadata: Dict = None):
    """Log user activity"""
    execute_db(
        '''INSERT INTO user_activity 
           (user_id, activity_type, description, ip_address, user_agent, metadata)
           VALUES (?, ?, ?, ?, ?, ?)''',
        [user_id, activity_type, description, request.remote_addr, 
         request.headers.get('User-Agent'), json.dumps(metadata) if metadata else None]
    )

def validate_email(email: str) -> bool:
    """Basic email validation"""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_username(username: str) -> bool:
    """Username validation"""
    import re
    pattern = r'^[a-zA-Z0-9_]{3,20}$'
    return re.match(pattern, username) is not None

# Route handlers
@app.route('/')
def index():
    """Serve main page"""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory(app.static_folder, path)

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'GoonSteen API is running'})

# Authentication endpoints
@app.route('/api/signup', methods=['POST'])
def signup():
    """User registration endpoint"""
    data = request.get_json()
    
    # Validate required fields
    required_fields = ['first_name', 'last_name', 'username', 'email', 'password', 'age_verification']
    for field in required_fields:
        if not data.get(field):
            return jsonify({'error': f'{field} is required'}), 400
    
    # Validate email format
    if not validate_email(data['email']):
        return jsonify({'error': 'Invalid email format'}), 400
    
    # Validate username format
    if not validate_username(data['username']):
        return jsonify({'error': 'Username must be 3-20 characters, letters and numbers only'}), 400
    
    # Check age verification (must be 18+)
    try:
        birth_date = datetime.strptime(data['age_verification'], '%Y-%m-%d').date()
        today = datetime.now().date()
        age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
        if age < 18:
            return jsonify({'error': 'You must be 18 or older to register'}), 400
    except ValueError:
        return jsonify({'error': 'Invalid birth date format'}), 400
    
    # Check if user already exists
    existing_user = query_db('SELECT id FROM users WHERE username = ? OR email = ?', 
                           [data['username'], data['email']], one=True)
    if existing_user:
        return jsonify({'error': 'Username or email already exists'}), 400
    
    # Hash password
    password_hash, salt = hash_password(data['password'])
    
    try:
        # Create user
        user_id = execute_db(
            '''INSERT INTO users 
               (username, email, password_hash, salt, first_name, last_name, date_of_birth,
                terms_accepted, marketing_emails, responsible_gambling)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            [data['username'], data['email'], password_hash, salt,
             data['first_name'], data['last_name'], data['age_verification'],
             data.get('terms_accepted', False), data.get('marketing_emails', False),
             data.get('responsible_gambling', False)]
        )
        
        # Create initial bankroll
        execute_db(
            'INSERT INTO bankrolls (user_id, total_balance, available_balance) VALUES (?, 1000.00, 1000.00)',
            [user_id]
        )
        
        # Log activity
        log_user_activity(user_id, 'user_registration', 'User account created')
        
        logger.info(f"New user registered: {data['username']} (ID: {user_id})")
        
        return jsonify({
            'message': 'Account created successfully',
            'user_id': user_id
        }), 201
        
    except sqlite3.Error as e:
        logger.error(f"Database error during signup: {e}")
        return jsonify({'error': 'Registration failed'}), 500

@app.route('/api/login', methods=['POST'])
def login():
    """User login endpoint"""
    data = request.get_json()
    
    username = data.get('username')
    password = data.get('password')
    remember_me = data.get('remember_me', False)
    
    if not username or not password:
        return jsonify({'error': 'Username and password are required'}), 400
    
    # Find user by username or email
    user = query_db(
        'SELECT * FROM users WHERE username = ? OR email = ?',
        [username, username], one=True
    )
    
    if not user:
        log_user_activity(None, 'login_failed', f'Login attempt with unknown username: {username}')
        return jsonify({'error': 'Invalid credentials'}), 401
    
    # Check if account is locked
    if user['locked_until'] and datetime.fromisoformat(user['locked_until']) > datetime.now(timezone.utc):
        return jsonify({'error': 'Account is temporarily locked'}), 423
    
    # Verify password
    if not verify_password(password, user['password_hash'], user['salt']):
        # Increment login attempts
        attempts = user['login_attempts'] + 1
        locked_until = None
        
        if attempts >= 5:
            locked_until = datetime.now(timezone.utc) + timedelta(minutes=30)
        
        execute_db(
            'UPDATE users SET login_attempts = ?, locked_until = ? WHERE id = ?',
            [attempts, locked_until.isoformat() if locked_until else None, user['id']]
        )
        
        log_user_activity(user['id'], 'login_failed', 'Invalid password')
        return jsonify({'error': 'Invalid credentials'}), 401
    
    # Check account status
    if user['status'] != 'active':
        log_user_activity(user['id'], 'login_failed', f'Login attempt with {user["status"]} account')
        return jsonify({'error': f'Account is {user["status"]}'}), 403
    
    # Reset login attempts on successful login
    execute_db(
        'UPDATE users SET login_attempts = 0, locked_until = NULL, last_login = ? WHERE id = ?',
        [datetime.now(timezone.utc).isoformat(), user['id']]
    )
    
    # Create session token
    token = create_jwt_token(user['id'])
    
    # Create session record
    session_token = generate_session_token()
    expires_at = datetime.now(timezone.utc) + timedelta(hours=24 if remember_me else 8)
    
    execute_db(
        '''INSERT INTO user_sessions 
           (user_id, session_token, ip_address, user_agent, expires_at)
           VALUES (?, ?, ?, ?, ?)''',
        [user['id'], session_token, request.remote_addr, 
         request.headers.get('User-Agent'), expires_at.isoformat()]
    )
    
    # Log successful login
    log_user_activity(user['id'], 'login_success', 'User logged in successfully')
    
    logger.info(f"User logged in: {user['username']} (ID: {user['id']})")
    
    return jsonify({
        'message': 'Login successful',
        'token': token,
        'user': {
            'id': user['id'],
            'username': user['username'],
            'email': user['email'],
            'first_name': user['first_name'],
            'last_name': user['last_name'],
            'is_admin': bool(user['is_admin']),
            'subscription_type': user['subscription_type']
        }
    })

@app.route('/api/logout', methods=['POST'])
@login_required
def logout():
    """User logout endpoint"""
    # Remove session
    session.clear()
    
    # Log activity
    log_user_activity(g.current_user['id'], 'logout', 'User logged out')
    
    return jsonify({'message': 'Logged out successfully'})

@app.route('/api/session', methods=['GET'])
@login_required
def get_session():
    """Get current session info"""
    return jsonify({
        'authenticated': True,
        'user': {
            'id': g.current_user['id'],
            'username': g.current_user['username'],
            'email': g.current_user['email'],
            'first_name': g.current_user['first_name'],
            'last_name': g.current_user['last_name'],
            'is_admin': bool(g.current_user['is_admin']),
            'subscription_type': g.current_user['subscription_type']
        }
    })

@app.route('/api/check-username', methods=['GET'])
def check_username():
    """Check username availability"""
    username = request.args.get('username')
    
    if not username or len(username) < 3:
        return jsonify({'available': False, 'error': 'Username too short'})
    
    existing_user = query_db('SELECT id FROM users WHERE username = ?', [username], one=True)
    
    return jsonify({'available': existing_user is None})

# Dashboard endpoints
@app.route('/api/dashboard/overview', methods=['GET'])
@login_required
def dashboard_overview():
    """Get dashboard overview data"""
    user_id = g.current_user['id']
    
    # Get bankroll info
    bankroll = query_db('SELECT * FROM bankrolls WHERE user_id = ?', [user_id], one=True)
    
    # Get betting stats
    bet_stats = query_db('''
        SELECT 
            COUNT(*) as total_bets,
            COUNT(CASE WHEN status = 'pending' THEN 1 END) as active_bets,
            COUNT(CASE WHEN status = 'won' THEN 1 END) as won_bets,
            COUNT(CASE WHEN status = 'lost' THEN 1 END) as lost_bets,
            SUM(CASE WHEN status = 'won' THEN actual_payout - stake ELSE 0 END) as total_profit,
            COUNT(CASE WHEN placed_at >= date('now', '-7 days') AND status = 'won' THEN 1 END) as week_wins,
            COUNT(CASE WHEN placed_at >= date('now', '-7 days') AND status = 'lost' THEN 1 END) as week_losses
        FROM bets 
        WHERE user_id = ?
    ''', [user_id], one=True)
    
    return jsonify({
        'bankroll': float(bankroll['total_balance']) if bankroll else 1000.00,
        'bankroll_change': 12.5,  # Calculate actual change
        'active_bets': bet_stats['active_bets'] or 0,
        'pending_results': 3,  # Calculate actual pending
        'week_wins': bet_stats['week_wins'] or 0,
        'week_losses': bet_stats['week_losses'] or 0,
        'profit_loss': float(bet_stats['total_profit']) if bet_stats['total_profit'] else 0.00,
        'roi': 15.3  # Calculate actual ROI
    })

@app.route('/api/dashboard/games', methods=['GET'])
@login_required
def dashboard_games():
    """Get today's games with predictions"""
    games = query_db('''
        SELECT 
            g.id, g.game_date, g.game_time, g.status,
            ht.name as home_team_name, ht.abbreviation as home_team_abbr,
            at.name as away_team_name, at.abbreviation as away_team_abbr,
            g.home_score, g.away_score,
            p.confidence, p.predicted_winner, p.predicted_home_score, p.predicted_away_score
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.id
        JOIN teams at ON g.away_team_id = at.id
        LEFT JOIN predictions p ON g.id = p.game_id AND p.model_name = 'Ensemble_NBA_v1'
        WHERE g.game_date = date('now')
        ORDER BY g.game_time
    ''')
    
    game_list = []
    for game in games:
        game_data = {
            'id': game['id'],
            'start_time': f"{game['game_date']} {game['game_time']}",
            'confidence': game['confidence'] or 75,
            'home_team': {
                'name': game['home_team_name'],
                'abbreviation': game['home_team_abbr'],
                'record': '25-15',  # Get from actual data
                'odds': '-110'
            },
            'away_team': {
                'name': game['away_team_name'],
                'abbreviation': game['away_team_abbr'],
                'record': '22-18',  # Get from actual data
                'odds': '+105'
            },
            'prediction': {
                'winner': game['home_team_name'] if game['predicted_winner'] else game['away_team_name'],
                'score': f"{int(game['predicted_home_score'] or 110)}-{int(game['predicted_away_score'] or 105)}",
                'total': '220.5',
                'over_under': 'Over'
            }
        }
        game_list.append(game_data)
    
    return jsonify(game_list)

@app.route('/api/dashboard/activity', methods=['GET'])
@login_required
def dashboard_activity():
    """Get recent user activity"""
    user_id = g.current_user['id']
    
    activities = query_db('''
        SELECT 
            'bet' as type,
            'Lakers vs Warriors' as title,
            'Lakers -3.5 • Won' as description,
            datetime('now', '-2 hours') as timestamp,
            125.00 as amount,
            'won' as status
        UNION ALL
        SELECT 
            'bet' as type,
            'Celtics vs Heat' as title,
            'Over 225.5 • Lost' as description,
            datetime('now', '-5 hours') as timestamp,
            -50.00 as amount,
            'lost' as status
        UNION ALL
        SELECT 
            'bet' as type,
            '3-Team Parlay' as title,
            '2/3 legs complete • Pending' as description,
            datetime('now', '-1 day') as timestamp,
            200.00 as amount,
            'pending' as status
        ORDER BY timestamp DESC
        LIMIT 10
    ''')
    
    return jsonify([dict(activity) for activity in activities])

# Admin endpoints
@app.route('/api/admin/overview', methods=['GET'])
@login_required
@admin_required
def admin_overview():
    """Get admin overview data"""
    # Get user statistics
    user_stats = query_db('''
        SELECT 
            COUNT(*) as total_users,
            COUNT(CASE WHEN created_at >= date('now', '-7 days') THEN 1 END) as new_users_week,
            COUNT(CASE WHEN status = 'active' THEN 1 END) as active_users,
            COUNT(CASE WHEN last_login >= date('now', '-30 days') THEN 1 END) as active_monthly
        FROM users
    ''', one=True)
    
    # Get betting statistics
    bet_stats = query_db('''
        SELECT 
            COUNT(*) as total_bets,
            COUNT(CASE WHEN placed_at >= date('now') THEN 1 END) as bets_today,
            AVG(CASE WHEN status IN ('won', 'lost') THEN 
                CASE WHEN status = 'won' THEN 1.0 ELSE 0.0 END 
            END) * 100 as win_rate
        FROM bets
    ''', one=True)
    
    return jsonify({
        'total_users': user_stats['total_users'] or 0,
        'new_users_week': user_stats['new_users_week'] or 0,
        'active_users_percentage': int((user_stats['active_monthly'] or 0) / max(user_stats['total_users'], 1) * 100),
        'total_bets': bet_stats['total_bets'] or 0,
        'bets_today': bet_stats['bets_today'] or 0,
        'win_rate': int(bet_stats['win_rate'] or 0),
        'revenue': 24650.00,  # Calculate from actual data
        'revenue_growth': 8.5,
        'revenue_target_percentage': 92,
        'model_accuracy': 68.9,
        'accuracy_improvement': 2.1
    })

@app.route('/api/admin/recent-users', methods=['GET'])
@login_required
@admin_required
def admin_recent_users():
    """Get recent users for admin"""
    users = query_db('''
        SELECT id, first_name, last_name, username, email, status, created_at
        FROM users 
        ORDER BY created_at DESC 
        LIMIT 10
    ''')
    
    return jsonify([dict(user) for user in users])

@app.route('/api/admin/activity', methods=['GET'])
@login_required
@admin_required
def admin_activity():
    """Get system activity for admin"""
    activities = query_db('''
        SELECT activity_type as type, description as title, description, created_at as timestamp
        FROM user_activity 
        ORDER BY created_at DESC 
        LIMIT 20
    ''')
    
    return jsonify([dict(activity) for activity in activities])

@app.route('/api/admin/users/<int:user_id>', methods=['GET'])
@login_required
@admin_required
def admin_get_user(user_id):
    """Get user details for admin"""
    user = query_db('SELECT * FROM users WHERE id = ?', [user_id], one=True)
    
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    # Get user stats
    stats = query_db('''
        SELECT 
            COUNT(*) as total_bets,
            AVG(CASE WHEN status IN ('won', 'lost') THEN 
                CASE WHEN status = 'won' THEN 1.0 ELSE 0.0 END 
            END) * 100 as win_rate,
            SUM(CASE WHEN status = 'won' THEN actual_payout - stake ELSE 0 END) as total_profit
        FROM bets 
        WHERE user_id = ?
    ''', [user_id], one=True)
    
    user_dict = dict(user)
    user_dict.update(dict(stats) if stats else {})
    
    return jsonify(user_dict)

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

# Database cleanup
@app.teardown_appcontext
def close_db_handler(error):
    close_db(error)

# CLI commands
@app.cli.command()
def init_database():
    """Initialize the database"""
    init_db()
    print("Database initialized successfully!")

@app.cli.command()
def create_admin():
    """Create admin user"""
    with app.app_context():
        username = input("Admin username: ")
        email = input("Admin email: ")
        password = input("Admin password: ")
        
        password_hash, salt = hash_password(password)
        
        try:
            user_id = execute_db(
                '''INSERT INTO users 
                   (username, email, password_hash, salt, first_name, last_name, 
                    date_of_birth, is_admin, email_verified, terms_accepted, responsible_gambling)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                [username, email, password_hash, salt, 'Admin', 'User', 
                 '1990-01-01', True, True, True, True]
            )
            print(f"Admin user created successfully with ID: {user_id}")
        except sqlite3.Error as e:
            print(f"Error creating admin user: {e}")

if __name__ == '__main__':
    # Initialize database if it doesn't exist
    if not os.path.exists(app.config['DATABASE']):
        with app.app_context():
            init_db()
    
    # Run the application
    app.run(
        host=os.environ.get('HOST', '127.0.0.1'),
        port=int(os.environ.get('PORT', 5000)),
        debug=os.environ.get('DEBUG', 'False').lower() == 'true'
    )
