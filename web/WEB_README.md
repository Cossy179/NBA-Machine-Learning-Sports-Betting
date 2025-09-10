
# GoonSteen Web Platform

A professional NBA sports betting platform with AI-powered predictions, built with modern web technologies.

## 🏀 Features

### User Features
- **Modern Landing Page** - Clean, responsive design with compelling call-to-action
- **User Authentication** - Secure signup/login with form validation and admin access
- **Interactive Dashboard** - Real-time NBA predictions and betting analytics
- **AI Predictions** - Integration with existing machine learning models
- **Parlay Builder** - Create optimized multi-game bets
- **Bankroll Management** - Track profits, losses, and betting limits
- **Responsive Design** - Optimized for desktop, tablet, and mobile devices

### Admin Features
- **Admin Dashboard** - Comprehensive user and system monitoring
- **User Management** - View, edit, suspend/unsuspend users
- **System Analytics** - Performance metrics and health monitoring
- **Activity Tracking** - Real-time user activity and system events
- **Model Management** - Monitor AI model performance and accuracy

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Modern web browser
- 2GB+ available disk space

### Installation

1. **Install Python Dependencies**
   ```bash
   pip install -r web_requirements.txt
   ```

2. **Initialize Database**
   ```bash
   python web_backend.py init-database
   ```

3. **Create Admin User**
   ```bash
   python web_backend.py create-admin
   ```

4. **Start the Server**
   ```bash
   python web_backend.py
   ```

5. **Access the Platform**
   - Open your browser to `http://localhost:5000`
   - Use the admin credentials you created to access admin features

## 📁 Project Structure

```
web/
├── index.html              # Landing page
├── login.html              # User login page
├── signup.html             # User registration page
├── dashboard.html          # User dashboard
├── admin-dashboard.html    # Admin dashboard
├── css/
│   ├── styles.css          # Main styles
│   ├── auth.css           # Authentication pages styles
│   ├── dashboard.css      # Dashboard styles
│   └── admin.css          # Admin dashboard styles
└── js/
    ├── main.js            # Core JavaScript functionality
    ├── auth.js            # Authentication logic
    ├── dashboard.js       # Dashboard functionality
    └── admin.js           # Admin dashboard logic

database_schema.sql         # SQLite database schema
web_backend.py             # Flask backend server
web_requirements.txt       # Python dependencies
```

## 🎨 Design Features

### Modern UI/UX
- **Clean Design** - Minimalist interface with intuitive navigation
- **Dark/Light Themes** - Automatic theme detection and manual toggle
- **Smooth Animations** - CSS transitions and JavaScript interactions
- **Mobile-First** - Responsive design that works on all devices

### Color Palette
- **Primary**: Blue gradient (#2563eb to #1d4ed8)
- **Secondary**: Amber (#f59e0b)
- **Success**: Emerald (#10b981)
- **Background**: Light gray (#f9fafb) with white cards

### Typography
- **Font Family**: Inter (Google Fonts)
- **Weights**: 300, 400, 500, 600, 700
- **Icons**: Font Awesome 6.4.0

## 🔐 Authentication System

### User Registration
- **Validation**: Email format, username uniqueness, password strength
- **Age Verification**: Must be 18+ to register
- **Terms & Conditions**: Required acceptance of terms and responsible gambling
- **Email Verification**: Token-based verification (ready for implementation)

### Login Security
- **JWT Tokens**: Secure authentication with expiration
- **Rate Limiting**: Protection against brute force attacks
- **Session Management**: Secure session handling with automatic cleanup
- **Admin Access**: Special admin login mode with elevated privileges

### Password Security
- **PBKDF2 Hashing**: Industry-standard password hashing
- **Salt Generation**: Unique salt for each password
- **Strength Validation**: Real-time password strength checking
- **Reset Functionality**: Secure password reset flow (ready for implementation)

## 📊 Database Schema

### Core Tables
- **users**: User accounts and profiles
- **user_sessions**: Active user sessions
- **user_activity**: Activity logging and audit trail
- **bankrolls**: User bankroll management
- **bankroll_transactions**: Financial transaction history

### Betting Tables
- **teams**: NBA team information
- **games**: Game schedules and results
- **betting_odds**: Sportsbook odds data
- **predictions**: AI model predictions
- **bets**: User betting history
- **bet_legs**: Parlay bet components

### System Tables
- **model_performance**: AI model tracking
- **system_settings**: Configuration management
- **notifications**: User notifications
- **system_logs**: System event logging
- **api_usage**: API usage tracking

## 🤖 AI Integration

### Model Integration Points
The platform is designed to integrate with your existing NBA prediction models:

1. **Prediction Pipeline**
   ```python
   # Integration point in web_backend.py
   from src.Predict.Advanced_Prediction_Runner import run_predictions
   
   @app.route('/api/predictions/update', methods=['POST'])
   def update_predictions():
       # Run your existing prediction models
       predictions = run_predictions()
       # Store in database
       store_predictions(predictions)
   ```

2. **Model Performance Tracking**
   - Automatic accuracy calculation
   - ROI and profit/loss tracking
   - Model comparison and selection

3. **Real-time Updates**
   - WebSocket integration for live updates
   - Automatic model retraining triggers
   - Performance monitoring alerts

## 🎯 Admin Features

### User Management
- **User Overview**: Total users, active users, new registrations
- **User Details**: Complete user profiles with betting statistics
- **Account Actions**: Suspend, unsuspend, edit user accounts
- **Activity Monitoring**: Real-time user activity tracking

### System Monitoring
- **Performance Metrics**: CPU, memory, database usage
- **System Health**: API response times, error rates
- **Model Status**: AI model performance and accuracy
- **Activity Feed**: System events and user actions

### Analytics Dashboard
- **Revenue Tracking**: Subscription revenue and growth metrics
- **Betting Analytics**: Win rates, popular bets, user behavior
- **Model Performance**: Accuracy trends and model comparison
- **User Engagement**: Activity patterns and retention metrics

## 🔧 Configuration

### Environment Variables
```bash
# Server Configuration
HOST=127.0.0.1
PORT=5000
DEBUG=False

# Database
DATABASE_PATH=web_database.db

# Security
SECRET_KEY=your-secret-key-here
JWT_SECRET=your-jwt-secret-here
JWT_EXPIRATION_HOURS=24

# Email (for future implementation)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
```

### System Settings
The platform includes a flexible settings system accessible via the database:

```sql
-- Example system settings
INSERT INTO system_settings (setting_key, setting_value, setting_type, description) VALUES
('max_login_attempts', '5', 'integer', 'Maximum login attempts before lockout'),
('session_timeout', '1440', 'integer', 'Session timeout in minutes'),
('default_bankroll', '1000.00', 'decimal', 'Default bankroll for new users'),
('kelly_criterion_enabled', 'true', 'boolean', 'Enable Kelly Criterion calculations');
```

## 🚀 Deployment

### Development Server
```bash
python web_backend.py
```

### Production Deployment
```bash
# Using Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 web_backend:app

# Using Waitress (Windows-friendly)
waitress-serve --port=5000 web_backend:app
```

### Docker Deployment (Optional)
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY web_requirements.txt .
RUN pip install -r web_requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "web_backend.py"]
```

## 📱 Mobile Optimization

### Responsive Breakpoints
- **Desktop**: 1200px+
- **Tablet**: 768px - 1199px
- **Mobile**: < 768px

### Mobile Features
- **Touch-Friendly**: Large tap targets and swipe gestures
- **Mobile Navigation**: Collapsible hamburger menu
- **Optimized Forms**: Mobile-friendly form inputs
- **Fast Loading**: Optimized assets and lazy loading

## 🔒 Security Features

### Data Protection
- **SQL Injection Prevention**: Parameterized queries
- **XSS Protection**: Input sanitization and CSP headers
- **CSRF Protection**: Token-based CSRF protection
- **Rate Limiting**: API endpoint rate limiting

### Privacy Compliance
- **Data Minimization**: Only collect necessary user data
- **Secure Storage**: Encrypted sensitive data storage
- **Audit Logging**: Comprehensive activity logging
- **Right to Deletion**: User account deletion capability

## 🧪 Testing

### Manual Testing Checklist
- [ ] User registration with validation
- [ ] Login/logout functionality
- [ ] Admin dashboard access
- [ ] Mobile responsiveness
- [ ] Form validations
- [ ] Error handling

### Automated Testing (Future)
```bash
# Run tests
pytest tests/

# Coverage report
pytest --cov=web_backend tests/
```

## 📈 Performance Optimization

### Frontend Optimization
- **Minified Assets**: Compressed CSS and JavaScript
- **Image Optimization**: WebP format with fallbacks
- **Caching**: Browser caching for static assets
- **CDN**: Font Awesome and Chart.js from CDN

### Backend Optimization
- **Database Indexing**: Optimized database queries
- **Connection Pooling**: Efficient database connections
- **Caching**: Redis caching for frequent queries (ready for implementation)
- **Async Processing**: Background task processing (ready for implementation)

## 🛠️ Maintenance

### Regular Tasks
- **Database Cleanup**: Remove old sessions and logs
- **Backup Management**: Regular database backups
- **Security Updates**: Keep dependencies updated
- **Performance Monitoring**: Monitor system health

### Monitoring Commands
```bash
# Database size
ls -lh web_database.db

# Active sessions
sqlite3 web_database.db "SELECT COUNT(*) FROM user_sessions WHERE expires_at > datetime('now');"

# System logs
tail -f system.log
```

## 🤝 Contributing

### Code Style
- **Python**: Follow PEP 8 guidelines
- **JavaScript**: Use ES6+ features
- **CSS**: BEM methodology for class naming
- **HTML**: Semantic HTML5 elements

### Git Workflow
```bash
# Feature branch
git checkout -b feature/new-feature
git commit -m "Add: new feature description"
git push origin feature/new-feature
```

## 📞 Support

### Common Issues
1. **Database Locked**: Restart the server
2. **Admin Access**: Ensure admin user is created
3. **Port Conflicts**: Change PORT environment variable
4. **Missing Dependencies**: Run `pip install -r web_requirements.txt`

### Contact Information
- **Developer**: [Your Name]
- **Email**: [Your Email]
- **Documentation**: This README file
- **Issues**: GitHub Issues (if using version control)

## 📝 License

This project is proprietary software. All rights reserved.

---

**Built with ❤️ for NBA sports betting enthusiasts**
