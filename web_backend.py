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

def hash_password(password: str, salt = None) -> tuple:
    """Hash password with salt"""
    if salt is None:
        salt = secrets.token_hex(16)
    
    # Handle both string and bytes salt
    if isinstance(salt, str):
        salt_bytes = salt.encode('utf-8')
    else:
        salt_bytes = salt
    
    password_hash = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt_bytes,
        100000
    )
    return password_hash, salt

def verify_password(password: str, password_hash, salt) -> bool:
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

@app.route('/api/admin/users/<int:user_id>/suspend', methods=['POST'])
@login_required
@admin_required
def suspend_user(user_id):
    """Suspend a user account"""
    try:
        execute_db('UPDATE users SET status = ? WHERE id = ?', ['suspended', user_id])
        log_user_activity(user_id, 'account_suspended', f'Account suspended by admin {g.current_user["username"]}')
        logger.info(f"User {user_id} suspended by admin {g.current_user['id']}")
        return jsonify({'message': 'User suspended successfully'})
    except sqlite3.Error as e:
        logger.error(f"Error suspending user {user_id}: {e}")
        return jsonify({'error': 'Failed to suspend user'}), 500

@app.route('/api/admin/users/<int:user_id>/unsuspend', methods=['POST'])
@login_required
@admin_required
def unsuspend_user(user_id):
    """Unsuspend a user account"""
    try:
        execute_db('UPDATE users SET status = ? WHERE id = ?', ['active', user_id])
        log_user_activity(user_id, 'account_unsuspended', f'Account unsuspended by admin {g.current_user["username"]}')
        logger.info(f"User {user_id} unsuspended by admin {g.current_user['id']}")
        return jsonify({'message': 'User unsuspended successfully'})
    except sqlite3.Error as e:
        logger.error(f"Error unsuspending user {user_id}: {e}")
        return jsonify({'error': 'Failed to unsuspend user'}), 500

@app.route('/api/admin/system-health', methods=['GET'])
@login_required
@admin_required
def system_health():
    """Get system health metrics"""
    import time
    
    try:
        # Try to import psutil for system metrics
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            has_psutil = True
        except ImportError:
            # Fallback values if psutil is not available
            cpu_percent = 23.0
            memory = type('Memory', (), {'percent': 67.0})()
            disk = type('Disk', (), {'percent': 12.0})()
            has_psutil = False
        
        # Database health check
        start_time = time.time()
        query_db('SELECT 1', one=True)
        db_response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Calculate database size
        db_size = os.path.getsize(app.config['DATABASE']) / (1024 * 1024)  # MB
        
        # Determine overall system status
        status = 'healthy'
        if cpu_percent > 85 or memory.percent > 90 or db_response_time > 1000:
            status = 'error'
        elif cpu_percent > 70 or memory.percent > 80 or db_response_time > 500:
            status = 'warning'
        
        return jsonify({
            'cpu': round(cpu_percent, 1),
            'memory': round(memory.percent, 1),
            'disk': round(disk.percent, 1) if has_psutil else 12.0,
            'database_size_mb': round(db_size, 2),
            'database_response_ms': round(db_response_time, 1),
            'api_response': round(db_response_time, 1),
            'status': status,
            'has_psutil': has_psutil
        })
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        return jsonify({
            'cpu': 0,
            'memory': 0,
            'disk': 0,
            'database_size_mb': 0,
            'database_response_ms': 0,
            'api_response': 0,
            'status': 'error',
            'has_psutil': False
        })

@app.route('/api/admin/chart-data', methods=['GET'])
@login_required
@admin_required
def admin_chart_data():
    """Get chart data for admin dashboard"""
    metric = request.args.get('metric', 'users')
    
    if metric == 'users':
        # Get user registration data for last 7 days
        data = query_db('''
            SELECT 
                date(created_at) as date,
                COUNT(*) as count
            FROM users 
            WHERE created_at >= date('now', '-7 days')
            GROUP BY date(created_at)
            ORDER BY date
        ''')
        
        labels = [row['date'] for row in data]
        values = [row['count'] for row in data]
        
    elif metric == 'bets':
        # Get betting data for last 7 days
        data = query_db('''
            SELECT 
                date(placed_at) as date,
                COUNT(*) as count
            FROM bets 
            WHERE placed_at >= date('now', '-7 days')
            GROUP BY date(placed_at)
            ORDER BY date
        ''')
        
        labels = [row['date'] for row in data]
        values = [row['count'] for row in data]
        
    elif metric == 'revenue':
        # Get revenue data for last 7 days
        data = query_db('''
            SELECT 
                date(us.created_at) as date,
                SUM(sp.price) as revenue
            FROM user_subscriptions us
            JOIN subscription_plans sp ON us.plan_id = sp.id
            WHERE us.created_at >= date('now', '-7 days')
            GROUP BY date(us.created_at)
            ORDER BY date
        ''')
        
        labels = [row['date'] for row in data]
        values = [float(row['revenue']) for row in data]
    
    else:
        labels = []
        values = []
    
    return jsonify({
        'labels': labels,
        'values': values
    })

@app.route('/api/admin/realtime', methods=['GET'])
@login_required
@admin_required
def admin_realtime():
    """Get real-time admin data"""
    # Get current active users (logged in within last hour)
    active_users = query_db('''
        SELECT COUNT(DISTINCT user_id) as count
        FROM user_sessions 
        WHERE last_activity >= datetime('now', '-1 hour')
    ''', one=True)
    
    # Get system health
    try:
        import psutil
        system_health = {
            'cpu': round(psutil.cpu_percent(), 1),
            'memory': round(psutil.virtual_memory().percent, 1),
            'disk': round(psutil.disk_usage('/').percent, 1),
            'api_response': 156  # This would be calculated from actual response times
        }
    except:
        system_health = {
            'cpu': 23,
            'memory': 67,
            'disk': 12,
            'api_response': 156
        }
    
    return jsonify({
        'active_users': active_users['count'] or 0,
        'system_health': system_health
    })

@app.route('/api/admin/all-users', methods=['GET'])
@login_required
@admin_required
def admin_all_users():
    """Get all users for admin management"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)
    status_filter = request.args.get('status', '')
    search_term = request.args.get('search', '')
    
    # Build query with filters
    where_clauses = []
    params = []
    
    if status_filter:
        where_clauses.append('status = ?')
        params.append(status_filter)
    
    if search_term:
        where_clauses.append('(username LIKE ? OR email LIKE ? OR first_name LIKE ? OR last_name LIKE ?)')
        search_pattern = f'%{search_term}%'
        params.extend([search_pattern, search_pattern, search_pattern, search_pattern])
    
    where_clause = 'WHERE ' + ' AND '.join(where_clauses) if where_clauses else ''
    
    users = query_db(f'''
        SELECT id, first_name, last_name, username, email, status, subscription_type, 
               created_at, last_login, email_verified
        FROM users 
        {where_clause}
        ORDER BY created_at DESC 
        LIMIT ? OFFSET ?
    ''', params + [per_page, (page - 1) * per_page])
    
    return jsonify([dict(user) for user in users])

@app.route('/api/admin/detailed-activity', methods=['GET'])
@login_required
@admin_required
def admin_detailed_activity():
    """Get detailed system activity for admin"""
    activity_type = request.args.get('type', '')
    limit = request.args.get('limit', 100, type=int)
    
    where_clause = 'WHERE activity_type = ?' if activity_type else ''
    params = [activity_type] if activity_type else []
    
    activities = query_db(f'''
        SELECT 
            ua.activity_type, ua.description, ua.created_at, ua.ip_address,
            u.username, u.first_name, u.last_name
        FROM user_activity ua
        LEFT JOIN users u ON ua.user_id = u.id
        {where_clause}
        ORDER BY ua.created_at DESC 
        LIMIT ?
    ''', params + [limit])
    
    return jsonify([dict(activity) for activity in activities])

@app.route('/api/admin/betting-analytics', methods=['GET'])
@login_required
@admin_required
def admin_betting_analytics():
    """Get betting analytics for admin"""
    analytics = query_db('''
        SELECT 
            SUM(stake) as total_volume,
            AVG(stake) as avg_stake,
            COUNT(*) as total_bets,
            AVG(CASE WHEN status IN ('won', 'lost') THEN 
                CASE WHEN status = 'won' THEN 1.0 ELSE 0.0 END 
            END) * 100 as win_rate,
            SUM(CASE WHEN status = 'won' THEN actual_payout - stake 
                     WHEN status = 'lost' THEN -stake 
                     ELSE 0 END) as total_profit,
            SUM(stake) as total_stakes
        FROM bets
    ''', one=True)
    
    profit_margin = 0
    if analytics['total_stakes'] and analytics['total_stakes'] > 0:
        profit_margin = (analytics['total_profit'] or 0) / analytics['total_stakes'] * 100
    
    return jsonify({
        'total_volume': float(analytics['total_volume'] or 0),
        'avg_stake': float(analytics['avg_stake'] or 0),
        'win_rate': round(analytics['win_rate'] or 0, 1),
        'profit_margin': round(profit_margin, 1),
        'total_bets': analytics['total_bets'] or 0
    })

@app.route('/api/admin/recent-bets', methods=['GET'])
@login_required
@admin_required
def admin_recent_bets():
    """Get recent bets for admin"""
    limit = request.args.get('limit', 50, type=int)
    
    bets = query_db('''
        SELECT 
            b.id, b.bet_type, b.stake, b.odds, b.status, b.actual_payout, b.placed_at,
            u.username, u.first_name, u.last_name
        FROM bets b
        JOIN users u ON b.user_id = u.id
        ORDER BY b.placed_at DESC
        LIMIT ?
    ''', [limit])
    
    return jsonify([dict(bet) for bet in bets])

@app.route('/api/admin/model-performance', methods=['GET'])
@login_required
@admin_required
def admin_model_performance():
    """Get model performance data for admin"""
    models = query_db('''
        SELECT 
            model_name as name,
            AVG(accuracy) as accuracy,
            SUM(total_predictions) as total_predictions,
            SUM(correct_predictions) as correct_predictions,
            AVG(roi) as roi
        FROM model_performance 
        WHERE date_from >= date('now', '-30 days')
        GROUP BY model_name
        ORDER BY accuracy DESC
    ''')
    
    model_data = []
    for model in models:
        model_dict = dict(model)
        model_dict['accuracy'] = round(model_dict['accuracy'] or 0, 1)
        model_dict['total_predictions'] = model_dict['total_predictions'] or 0
        model_dict['roi'] = round(model_dict['roi'] or 0, 2)
        model_data.append(model_dict)
    
    return jsonify(model_data)

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

def validate_admin_password(password):
    """Validate admin password strength"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    weak_passwords = ['admin', 'admin123', 'password', '123456', 'password123', 'qwerty', 'abc123']
    if password.lower() in weak_passwords:
        return False, "Password is too weak/common"
    
    # Check for at least one uppercase, lowercase, digit, and special character
    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)
    has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
    
    if not (has_upper and has_lower and has_digit and has_special):
        return False, "Password must contain uppercase, lowercase, digit, and special character"
    
    return True, "Password is strong"

@app.cli.command()
def create_admin():
    """Create admin user with strong password validation"""
    with app.app_context():
        print("🔑 Creating Admin User")
        print("=" * 30)
        
        username = input("Admin username: ")
        email = input("Admin email: ")
        
        # Password validation loop
        while True:
            password = input("Admin password: ")
            is_valid, message = validate_admin_password(password)
            
            if is_valid:
                print(f"✅ {message}")
                break
            else:
                print(f"❌ {message}")
                print("Please try again with a stronger password.")
        
        # Confirm password
        confirm_password = input("Confirm password: ")
        if password != confirm_password:
            print("❌ Passwords do not match!")
            return
        
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
            print(f"✅ Admin user created successfully with ID: {user_id}")
            print(f"   Username: {username}")
            print(f"   Email: {email}")
        except sqlite3.Error as e:
            print(f"❌ Error creating admin user: {e}")

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
