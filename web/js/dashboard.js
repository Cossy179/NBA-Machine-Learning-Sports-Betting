// Dashboard JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Check if user is authenticated first
    if (!isAuthenticated()) {
        window.location.href = 'login.html';
        return;
    }
    
    initDashboard();
    initSidebar();
    initNotifications();
    initCharts();
    initRealTimeUpdates();
    checkAuthentication();
});

// Initialize dashboard
function initDashboard() {
    loadUserData();
    loadDashboardData();
    initPageNavigation();
    initMobileNavigation();
}

// Check if user is authenticated
async function checkAuthentication() {
    try {
        const response = await makeAuthenticatedRequest('/api/session');
        const data = await response.json();
        
        if (!data.authenticated) {
            window.location.href = 'login.html';
            return;
        }
        
        // Update user info in sidebar
        updateUserInfo(data.user);
        
        // If admin, show admin features
        if (data.user.is_admin) {
            showAdminFeatures();
        }
        
    } catch (error) {
        console.error('Authentication check failed:', error);
        window.location.href = 'login.html';
    }
}

// Update user info in sidebar
function updateUserInfo(user) {
    const userName = document.querySelector('.user-name');
    const userStatus = document.querySelector('.user-status');
    const userAvatar = document.querySelector('.user-avatar');
    
    if (userName) {
        userName.textContent = `${user.first_name} ${user.last_name}`;
    }
    
    if (userStatus) {
        userStatus.textContent = user.subscription_type || 'Free Member';
    }
    
    if (userAvatar && user.avatar) {
        userAvatar.innerHTML = `<img src="${user.avatar}" alt="Avatar">`;
    } else if (userAvatar) {
        userAvatar.innerHTML = user.first_name.charAt(0).toUpperCase();
    }
}

// Show admin features
function showAdminFeatures() {
    const sidebar = document.querySelector('.sidebar-nav');
    const adminItem = document.createElement('li');
    adminItem.className = 'nav-item';
    adminItem.innerHTML = `
        <a href="admin-dashboard.html" class="nav-link">
            <i class="fas fa-user-shield"></i>
            <span>Admin Panel</span>
        </a>
    `;
    sidebar.querySelector('.nav-list').appendChild(adminItem);
}

// Load user data
async function loadUserData() {
    try {
        const response = await makeAuthenticatedRequest('/api/user/profile');
        const userData = await response.json();
        
        if (response.ok) {
            updateDashboardWithUserData(userData);
        }
    } catch (error) {
        console.error('Failed to load user data:', error);
    }
}

// Load dashboard data
async function loadDashboardData() {
    try {
        const [overviewResponse, gamesResponse, activityResponse] = await Promise.all([
            makeAuthenticatedRequest('/api/dashboard/overview'),
            makeAuthenticatedRequest('/api/dashboard/games'),
            makeAuthenticatedRequest('/api/dashboard/activity')
        ]);
        
        const [overview, games, activity] = await Promise.all([
            overviewResponse.json(),
            gamesResponse.json(),
            activityResponse.json()
        ]);
        
        updateOverviewCards(overview);
        updateGamesList(games);
        updateActivityList(activity);
        
    } catch (error) {
        console.error('Failed to load dashboard data:', error);
        showErrorState();
    }
}

// Update overview cards
function updateOverviewCards(data) {
    const cards = document.querySelectorAll('.overview-card');
    
    cards.forEach((card, index) => {
        const value = card.querySelector('.card-value');
        const change = card.querySelector('.card-change');
        
        switch (index) {
            case 0: // Bankroll
                value.textContent = formatCurrency(data.bankroll);
                updateChangeIndicator(change, data.bankroll_change);
                break;
            case 1: // Active Bets
                value.textContent = data.active_bets;
                change.innerHTML = `<i class="fas fa-clock"></i> ${data.pending_results} pending results`;
                break;
            case 2: // This Week
                value.textContent = `${data.week_wins}-${data.week_losses}`;
                const winRate = (data.week_wins / (data.week_wins + data.week_losses) * 100).toFixed(1);
                updateChangeIndicator(change, winRate, '%', 'win rate');
                break;
            case 3: // Profit/Loss
                value.textContent = formatCurrency(data.profit_loss);
                updateChangeIndicator(change, data.roi, '%', 'ROI');
                break;
        }
    });
}

// Update games list
function updateGamesList(games) {
    const gamesList = document.querySelector('.games-list');
    if (!gamesList) return;
    
    gamesList.innerHTML = '';
    
    games.forEach(game => {
        const gameCard = createGameCard(game);
        gamesList.appendChild(gameCard);
    });
}

// Create game card
function createGameCard(game) {
    const card = document.createElement('div');
    card.className = 'game-card';
    
    const confidenceClass = getConfidenceClass(game.confidence);
    
    card.innerHTML = `
        <div class="game-header">
            <div class="game-time">${formatTime(game.start_time)}</div>
            <div class="confidence-badge ${confidenceClass}">${game.confidence}% Confidence</div>
        </div>
        <div class="game-matchup">
            <div class="team away">
                <div class="team-logo">${game.away_team.abbreviation}</div>
                <div class="team-info">
                    <div class="team-name">${game.away_team.name}</div>
                    <div class="team-record">${game.away_team.record}</div>
                </div>
                <div class="team-odds">${game.away_team.odds}</div>
            </div>
            <div class="vs">@</div>
            <div class="team home">
                <div class="team-logo">${game.home_team.abbreviation}</div>
                <div class="team-info">
                    <div class="team-name">${game.home_team.name}</div>
                    <div class="team-record">${game.home_team.record}</div>
                </div>
                <div class="team-odds">${game.home_team.odds}</div>
            </div>
        </div>
        <div class="game-prediction">
            <div class="prediction-result">
                <i class="fas fa-trophy"></i>
                ${game.prediction.winner} Win
            </div>
            <div class="prediction-details">
                <span>Predicted Score: ${game.prediction.score}</span>
                <span>O/U: ${game.prediction.total} (${game.prediction.over_under})</span>
            </div>
        </div>
        <div class="game-actions">
            <button class="btn btn-outline btn-sm" onclick="viewGameAnalysis('${game.id}')">View Analysis</button>
            <button class="btn btn-primary btn-sm" onclick="placeBet('${game.id}')">Place Bet</button>
        </div>
    `;
    
    return card;
}

// Update activity list
function updateActivityList(activities) {
    const activityList = document.querySelector('.activity-list');
    if (!activityList) return;
    
    activityList.innerHTML = '';
    
    activities.slice(0, 5).forEach(activity => {
        const activityItem = createActivityItem(activity);
        activityList.appendChild(activityItem);
    });
}

// Create activity item
function createActivityItem(activity) {
    const item = document.createElement('div');
    item.className = 'activity-item';
    
    const iconClass = getActivityIconClass(activity.status);
    const amountClass = getAmountClass(activity.amount, activity.status);
    
    item.innerHTML = `
        <div class="activity-icon ${iconClass}">
            <i class="fas ${getActivityIcon(activity.status)}"></i>
        </div>
        <div class="activity-content">
            <div class="activity-title">${activity.title}</div>
            <div class="activity-description">${activity.description}</div>
            <div class="activity-time">${formatRelativeTime(activity.timestamp)}</div>
        </div>
        <div class="activity-amount ${amountClass}">${formatActivityAmount(activity.amount, activity.status)}</div>
    `;
    
    return item;
}

// Sidebar functionality
function initSidebar() {
    const sidebarToggle = document.getElementById('sidebarToggle');
    const sidebar = document.querySelector('.sidebar');
    
    if (sidebarToggle) {
        sidebarToggle.addEventListener('click', function() {
            sidebar.classList.toggle('collapsed');
        });
    }
    
    // Handle mobile sidebar
    const mobileToggle = document.querySelector('.mobile-sidebar-toggle');
    if (mobileToggle) {
        mobileToggle.addEventListener('click', function() {
            sidebar.classList.toggle('mobile-visible');
        });
    }
    
    // Close sidebar on mobile when clicking outside
    document.addEventListener('click', function(e) {
        if (window.innerWidth <= 1024) {
            if (!sidebar.contains(e.target) && !e.target.closest('.mobile-sidebar-toggle')) {
                sidebar.classList.remove('mobile-visible');
            }
        }
    });
}

// Page navigation
function initPageNavigation() {
    const navLinks = document.querySelectorAll('.nav-link[data-page]');
    
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            const page = this.getAttribute('data-page');
            navigateToPage(page);
            
            // Update active state
            document.querySelectorAll('.nav-item').forEach(item => item.classList.remove('active'));
            this.closest('.nav-item').classList.add('active');
        });
    });
}

// Navigate to page
function navigateToPage(page) {
    const content = document.getElementById('dashboardContent');
    const pageTitle = document.querySelector('.page-title');
    const pageSubtitle = document.querySelector('.page-subtitle');
    
    // Update URL without reload
    history.pushState({page}, '', `#${page}`);
    
    // Update page title and content
    switch (page) {
        case 'predictions':
            pageTitle.textContent = 'AI Predictions';
            pageSubtitle.textContent = 'Advanced machine learning predictions for NBA games';
            loadPredictionsPage();
            break;
        case 'parlays':
            pageTitle.textContent = 'Parlay Builder';
            pageSubtitle.textContent = 'Build optimized parlay bets with correlation analysis';
            loadParlaysPage();
            break;
        case 'analytics':
            pageTitle.textContent = 'Analytics';
            pageSubtitle.textContent = 'Detailed performance analytics and insights';
            loadAnalyticsPage();
            break;
        case 'history':
            pageTitle.textContent = 'Bet History';
            pageSubtitle.textContent = 'Complete history of your betting activity';
            loadHistoryPage();
            break;
        case 'bankroll':
            pageTitle.textContent = 'Bankroll Management';
            pageSubtitle.textContent = 'Manage your betting bankroll and set limits';
            loadBankrollPage();
            break;
        case 'settings':
            pageTitle.textContent = 'Settings';
            pageSubtitle.textContent = 'Customize your account and preferences';
            loadSettingsPage();
            break;
        default:
            pageTitle.textContent = 'Dashboard';
            pageSubtitle.textContent = 'Welcome back! Here\'s your betting overview.';
            loadDashboardPage();
    }
}

// Mobile navigation
function initMobileNavigation() {
    const mobileNavItems = document.querySelectorAll('.mobile-nav-item[data-page]');
    
    mobileNavItems.forEach(item => {
        item.addEventListener('click', function(e) {
            e.preventDefault();
            
            const page = this.getAttribute('data-page');
            navigateToPage(page);
            
            // Update active state
            document.querySelectorAll('.mobile-nav-item').forEach(nav => nav.classList.remove('active'));
            this.classList.add('active');
        });
    });
}

// Notifications
function initNotifications() {
    const notificationsBtn = document.querySelector('.notifications-btn');
    const notificationPanel = document.getElementById('notificationPanel');
    
    if (notificationsBtn) {
        notificationsBtn.addEventListener('click', function() {
            notificationPanel.classList.toggle('active');
            loadNotifications();
        });
    }
    
    // Close notifications when clicking outside
    document.addEventListener('click', function(e) {
        if (!notificationPanel.contains(e.target) && !notificationsBtn.contains(e.target)) {
            notificationPanel.classList.remove('active');
        }
    });
}

// Load notifications
async function loadNotifications() {
    try {
        const response = await makeAuthenticatedRequest('/api/notifications');
        const notifications = await response.json();
        
        updateNotificationsList(notifications);
        updateNotificationBadge(notifications.filter(n => !n.read).length);
        
    } catch (error) {
        console.error('Failed to load notifications:', error);
    }
}

// Update notifications list
function updateNotificationsList(notifications) {
    const notificationList = document.querySelector('.notification-list');
    if (!notificationList) return;
    
    notificationList.innerHTML = '';
    
    notifications.forEach(notification => {
        const notificationItem = createNotificationItem(notification);
        notificationList.appendChild(notificationItem);
    });
}

// Create notification item
function createNotificationItem(notification) {
    const item = document.createElement('div');
    item.className = `notification-item ${!notification.read ? 'unread' : ''}`;
    
    item.innerHTML = `
        <div class="notification-icon">
            <i class="fas ${getNotificationIcon(notification.type)}"></i>
        </div>
        <div class="notification-content">
            <div class="notification-title">${notification.title}</div>
            <div class="notification-message">${notification.message}</div>
            <div class="notification-time">${formatRelativeTime(notification.timestamp)}</div>
        </div>
    `;
    
    item.addEventListener('click', function() {
        markNotificationAsRead(notification.id);
        if (notification.action_url) {
            window.location.href = notification.action_url;
        }
    });
    
    return item;
}

// Update notification badge
function updateNotificationBadge(count) {
    const badge = document.querySelector('.notification-badge');
    if (!badge) return;
    
    if (count > 0) {
        badge.textContent = count > 99 ? '99+' : count;
        badge.style.display = 'block';
    } else {
        badge.style.display = 'none';
    }
}

// Charts
function initCharts() {
    initPerformanceChart();
}

// Initialize performance chart
function initPerformanceChart() {
    const ctx = document.getElementById('performanceChart');
    if (!ctx) return;
    
    const chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul'],
            datasets: [{
                label: 'Bankroll',
                data: [1000, 1150, 1050, 1300, 1250, 1400, 1450],
                borderColor: 'rgb(37, 99, 235)',
                backgroundColor: 'rgba(37, 99, 235, 0.1)',
                fill: true,
                tension: 0.4
            }, {
                label: 'Profit/Loss',
                data: [0, 150, 50, 300, 250, 400, 450],
                borderColor: 'rgb(16, 185, 129)',
                backgroundColor: 'rgba(16, 185, 129, 0.1)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'bottom'
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                },
                x: {
                    grid: {
                        display: false
                    }
                }
            },
            elements: {
                point: {
                    radius: 4,
                    hoverRadius: 6
                }
            }
        }
    });
    
    // Chart controls
    const chartControls = document.querySelectorAll('.chart-control');
    chartControls.forEach(control => {
        control.addEventListener('click', function() {
            chartControls.forEach(c => c.classList.remove('active'));
            this.classList.add('active');
            
            const period = this.getAttribute('data-period');
            updateChartData(chart, period);
        });
    });
}

// Update chart data
async function updateChartData(chart, period) {
    try {
        const response = await makeAuthenticatedRequest(`/api/dashboard/chart-data?period=${period}`);
        const data = await response.json();
        
        chart.data.labels = data.labels;
        chart.data.datasets[0].data = data.bankroll;
        chart.data.datasets[1].data = data.profit_loss;
        chart.update();
        
    } catch (error) {
        console.error('Failed to update chart data:', error);
    }
}

// Real-time updates
function initRealTimeUpdates() {
    // Update every 30 seconds
    setInterval(updateRealTimeData, 30000);
    
    // Listen for WebSocket updates if available
    if (window.WebSocket) {
        connectWebSocket();
    }
}

// Update real-time data
async function updateRealTimeData() {
    try {
        const response = await makeAuthenticatedRequest('/api/dashboard/realtime');
        const data = await response.json();
        
        updateLiveScores(data.live_scores);
        updateActiveBets(data.active_bets);
        updateNotificationBadge(data.unread_notifications);
        
    } catch (error) {
        console.error('Failed to update real-time data:', error);
    }
}

// WebSocket connection
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/dashboard`);
    
    ws.onmessage = function(event) {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
    };
    
    ws.onclose = function() {
        // Reconnect after 5 seconds
        setTimeout(connectWebSocket, 5000);
    };
}

// Handle WebSocket messages
function handleWebSocketMessage(data) {
    switch (data.type) {
        case 'game_update':
            updateGameStatus(data.game_id, data.status);
            break;
        case 'bet_result':
            showBetResultNotification(data.bet);
            updateActivityList([data.bet, ...currentActivity]);
            break;
        case 'new_prediction':
            showNewPredictionNotification(data.prediction);
            break;
    }
}

// Utility functions
function formatCurrency(amount) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(amount);
}

function formatTime(timestamp) {
    return new Date(timestamp).toLocaleTimeString('en-US', {
        hour: 'numeric',
        minute: '2-digit',
        hour12: true
    });
}

function formatRelativeTime(timestamp) {
    const now = new Date();
    const time = new Date(timestamp);
    const diff = now - time;
    
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);
    
    if (minutes < 60) {
        return `${minutes} minute${minutes !== 1 ? 's' : ''} ago`;
    } else if (hours < 24) {
        return `${hours} hour${hours !== 1 ? 's' : ''} ago`;
    } else {
        return `${days} day${days !== 1 ? 's' : ''} ago`;
    }
}

function getConfidenceClass(confidence) {
    if (confidence >= 80) return 'high';
    if (confidence >= 60) return 'medium';
    return 'low';
}

function getActivityIconClass(status) {
    switch (status) {
        case 'won': return 'win';
        case 'lost': return 'loss';
        case 'pending': return 'pending';
        default: return 'pending';
    }
}

function getActivityIcon(status) {
    switch (status) {
        case 'won': return 'fa-check';
        case 'lost': return 'fa-times';
        case 'pending': return 'fa-clock';
        default: return 'fa-clock';
    }
}

function getAmountClass(amount, status) {
    if (status === 'pending') return 'pending';
    return amount > 0 ? 'positive' : 'negative';
}

function formatActivityAmount(amount, status) {
    if (status === 'pending') {
        return formatCurrency(Math.abs(amount));
    }
    return (amount > 0 ? '+' : '') + formatCurrency(amount);
}

function getNotificationIcon(type) {
    switch (type) {
        case 'bet_won': return 'fa-trophy';
        case 'bet_lost': return 'fa-times-circle';
        case 'new_prediction': return 'fa-brain';
        case 'bankroll_alert': return 'fa-exclamation-triangle';
        default: return 'fa-bell';
    }
}

function updateChangeIndicator(element, value, suffix = '%', label = '') {
    const isPositive = value > 0;
    const icon = isPositive ? 'fa-arrow-up' : 'fa-arrow-down';
    const className = isPositive ? 'positive' : 'negative';
    
    element.className = `card-change ${className}`;
    element.innerHTML = `
        <i class="fas ${icon}"></i>
        ${isPositive ? '+' : ''}${value}${suffix} ${label}
    `;
}

// Page-specific loaders
function loadDashboardPage() {
    // Already loaded in initDashboard
}

function loadPredictionsPage() {
    // Load predictions page content
    console.log('Loading predictions page...');
}

function loadParlaysPage() {
    // Load parlays page content
    console.log('Loading parlays page...');
}

function loadAnalyticsPage() {
    // Load analytics page content
    console.log('Loading analytics page...');
}

function loadHistoryPage() {
    // Load history page content
    console.log('Loading history page...');
}

function loadBankrollPage() {
    // Load bankroll page content
    console.log('Loading bankroll page...');
}

function loadSettingsPage() {
    // Load settings page content
    console.log('Loading settings page...');
}

// Action handlers
function viewGameAnalysis(gameId) {
    console.log('Viewing analysis for game:', gameId);
    // Implement game analysis modal or page
}

function placeBet(gameId) {
    console.log('Placing bet for game:', gameId);
    // Implement bet placement modal or page
}

function markNotificationAsRead(notificationId) {
    makeAuthenticatedRequest(`/api/notifications/${notificationId}/read`, {
        method: 'POST'
    }).then(() => {
        loadNotifications();
    });
}

function closeNotifications() {
    document.getElementById('notificationPanel').classList.remove('active');
}

function logout() {
    clearAuthData();
    makeAuthenticatedRequest('/api/logout', {
        method: 'POST'
    }).then(() => {
        window.location.href = 'login.html';
    }).catch(() => {
        // Even if logout fails, clear local data and redirect
        window.location.href = 'login.html';
    });
}

// Error state
function showErrorState() {
    const content = document.getElementById('dashboardContent');
    content.innerHTML = `
        <div class="error-state">
            <div class="error-icon">
                <i class="fas fa-exclamation-triangle"></i>
            </div>
            <h3>Unable to Load Dashboard</h3>
            <p>There was an error loading your dashboard data. Please try refreshing the page.</p>
            <button class="btn btn-primary" onclick="location.reload()">Refresh Page</button>
        </div>
    `;
}

// Export functions for global use
window.DashboardUtils = {
    navigateToPage,
    viewGameAnalysis,
    placeBet,
    logout,
    closeNotifications
};
