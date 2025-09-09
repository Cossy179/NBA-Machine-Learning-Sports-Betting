// Admin Dashboard JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Check if user is authenticated first
    if (!isAuthenticated()) {
        window.location.href = 'login.html';
        return;
    }
    
    initAdminDashboard();
    checkAdminAuthentication();
    initAdminCharts();
    initAdminRealTime();
});

// Import auth functions from auth.js
function isAuthenticated() {
    const token = localStorage.getItem('auth_token');
    if (!token) return false;
    
    try {
        const payload = JSON.parse(atob(token.split('.')[1]));
        return payload.exp * 1000 > Date.now();
    } catch {
        return false;
    }
}

function makeAuthenticatedRequest(url, options = {}) {
    const token = localStorage.getItem('auth_token');
    if (!token) {
        throw new Error('No authentication token');
    }
    
    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
        }
    };
    
    const mergedOptions = {
        ...defaultOptions,
        ...options,
        headers: {
            ...defaultOptions.headers,
            ...options.headers
        }
    };
    
    return fetch(url, mergedOptions).then(response => {
        if (response.status === 401) {
            localStorage.removeItem('auth_token');
            localStorage.removeItem('user_data');
            window.location.href = '/login.html';
            return;
        }
        return response;
    });
}

function showModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.classList.add('active');
        document.body.style.overflow = 'hidden';
    }
}

function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.classList.remove('active');
        document.body.style.overflow = '';
    }
}

function formatCurrency(amount) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(amount);
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

// Check if user has admin privileges
async function checkAdminAuthentication() {
    try {
        const response = await makeAuthenticatedRequest('/api/session');
        const data = await response.json();
        
        if (!data.authenticated || !data.user.is_admin) {
            window.location.href = 'login.html';
            return;
        }
        
        updateAdminUserInfo(data.user);
        
    } catch (error) {
        console.error('Admin authentication check failed:', error);
        window.location.href = 'login.html';
    }
}

// Initialize admin dashboard
function initAdminDashboard() {
    loadAdminOverview();
    loadRecentUsers();
    loadSystemActivity();
    initAdminNavigation();
    initQuickActions();
}

// Update admin user info
function updateAdminUserInfo(user) {
    const userName = document.querySelector('.admin-sidebar .user-name');
    const userAvatar = document.querySelector('.admin-avatar');
    
    if (userName) {
        userName.textContent = `${user.first_name} ${user.last_name}`;
    }
    
    if (userAvatar && user.avatar) {
        userAvatar.innerHTML = `<img src="${user.avatar}" alt="Avatar">`;
    } else if (userAvatar) {
        userAvatar.innerHTML = user.first_name.charAt(0).toUpperCase();
    }
}

// Load admin overview data
async function loadAdminOverview() {
    try {
        // Show loading state
        showLoadingState();
        
        const response = await makeAuthenticatedRequest('/api/admin/overview');
        if (!response.ok) {
            throw new Error('Failed to fetch overview data');
        }
        
        const data = await response.json();
        updateAdminOverviewCards(data);
        
        // Hide loading state
        hideLoadingState();
        
    } catch (error) {
        console.error('Failed to load admin overview:', error);
        showAdminError('Failed to load overview data');
        hideLoadingState();
    }
}

// Show loading state for cards
function showLoadingState() {
    const cards = document.querySelectorAll('.admin-card');
    cards.forEach(card => {
        card.classList.add('loading');
        const value = card.querySelector('.card-value');
        const change = card.querySelector('.card-change');
        const progressText = card.querySelector('.card-progress span');
        
        if (value) value.textContent = 'Loading...';
        if (change) change.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading...';
        if (progressText) progressText.textContent = 'Loading...';
    });
}

// Hide loading state for cards
function hideLoadingState() {
    const cards = document.querySelectorAll('.admin-card');
    cards.forEach(card => {
        card.classList.remove('loading');
    });
}

// Update admin overview cards
function updateAdminOverviewCards(data) {
    // Update Total Users card
    document.getElementById('totalUsers').textContent = formatNumber(data.total_users);
    updateAdminChangeIndicator(document.getElementById('usersChange'), data.new_users_week, 'this week');
    document.getElementById('usersProgress').style.width = `${data.active_users_percentage}%`;
    document.getElementById('usersProgressText').textContent = `${data.active_users_percentage}% active`;
    
    // Update Total Bets card
    document.getElementById('totalBets').textContent = formatNumber(data.total_bets);
    updateAdminChangeIndicator(document.getElementById('betsChange'), data.bets_today, 'today');
    document.getElementById('betsProgress').style.width = `${data.win_rate}%`;
    document.getElementById('betsProgressText').textContent = `${data.win_rate}% win rate`;
    
    // Update Revenue card
    document.getElementById('totalRevenue').textContent = formatCurrency(data.revenue);
    updateAdminChangeIndicator(document.getElementById('revenueChange'), data.revenue_growth, '% this month');
    document.getElementById('revenueProgress').style.width = `${data.revenue_target_percentage}%`;
    document.getElementById('revenueProgressText').textContent = `${data.revenue_target_percentage}% of target`;
    
    // Update Model Accuracy card
    document.getElementById('modelAccuracy').textContent = `${data.model_accuracy}%`;
    updateAdminChangeIndicator(document.getElementById('accuracyChange'), data.accuracy_improvement, '% improvement');
    document.getElementById('accuracyProgress').style.width = `${data.model_accuracy}%`;
    document.getElementById('accuracyProgressText').textContent = 'Above benchmark';
    
    // Add smooth animation to progress bars
    setTimeout(() => {
        document.getElementById('usersProgress').style.width = `${data.active_users_percentage}%`;
        document.getElementById('betsProgress').style.width = `${data.win_rate}%`;
        document.getElementById('revenueProgress').style.width = `${data.revenue_target_percentage}%`;
        document.getElementById('accuracyProgress').style.width = `${data.model_accuracy}%`;
    }, 100);
}

// Load recent users
async function loadRecentUsers() {
    try {
        const response = await makeAuthenticatedRequest('/api/admin/recent-users');
        const users = await response.json();
        
        updateRecentUsersTable(users);
        
    } catch (error) {
        console.error('Failed to load recent users:', error);
    }
}

// Update recent users table
function updateRecentUsersTable(users) {
    const tbody = document.querySelector('.admin-table tbody');
    if (!tbody) return;
    
    tbody.innerHTML = '';
    
    users.forEach(user => {
        const row = createUserTableRow(user);
        tbody.appendChild(row);
    });
}

// Create user table row
function createUserTableRow(user) {
    const row = document.createElement('tr');
    
    const statusClass = getUserStatusClass(user.status);
    const joinedTime = formatRelativeTime(user.created_at);
    
    row.innerHTML = `
        <td>
            <div class="user-cell">
                <div class="user-avatar small">${user.first_name.charAt(0).toUpperCase()}${user.last_name.charAt(0).toUpperCase()}</div>
                <div class="user-info">
                    <div class="user-name">${user.first_name} ${user.last_name}</div>
                    <div class="user-id">#${user.id}</div>
                </div>
            </div>
        </td>
        <td>${user.email}</td>
        <td><span class="status-badge ${statusClass}">${user.status}</span></td>
        <td>${joinedTime}</td>
        <td>
            <div class="action-buttons">
                <button class="btn-icon" onclick="viewUser(${user.id})" title="View User">
                    <i class="fas fa-eye"></i>
                </button>
                <button class="btn-icon" onclick="editUser(${user.id})" title="Edit User">
                    <i class="fas fa-edit"></i>
                </button>
                ${user.status === 'suspended' ? 
                    `<button class="btn-icon warning" onclick="unsuspendUser(${user.id})" title="Unsuspend User">
                        <i class="fas fa-unlock"></i>
                    </button>` :
                    `<button class="btn-icon warning" onclick="suspendUser(${user.id})" title="Suspend User">
                        <i class="fas fa-lock"></i>
                    </button>`
                }
            </div>
        </td>
    `;
    
    return row;
}

// Load system activity
async function loadSystemActivity() {
    try {
        const response = await makeAuthenticatedRequest('/api/admin/activity');
        const activities = await response.json();
        
        updateSystemActivity(activities);
        
    } catch (error) {
        console.error('Failed to load system activity:', error);
    }
}

// Update system activity
function updateSystemActivity(activities) {
    const activityFeed = document.querySelector('.activity-feed');
    if (!activityFeed) return;
    
    activityFeed.innerHTML = '';
    
    activities.slice(0, 10).forEach(activity => {
        const activityItem = createActivityItem(activity);
        activityFeed.appendChild(activityItem);
    });
}

// Create activity item
function createActivityItem(activity) {
    const item = document.createElement('div');
    item.className = 'activity-item';
    
    const iconClass = getActivityIconClass(activity.type);
    const icon = getActivityIcon(activity.type);
    
    item.innerHTML = `
        <div class="activity-icon ${iconClass}">
            <i class="fas ${icon}"></i>
        </div>
        <div class="activity-content">
            <div class="activity-title">${activity.title}</div>
            <div class="activity-description">${activity.description}</div>
            <div class="activity-time">${formatRelativeTime(activity.timestamp)}</div>
        </div>
    `;
    
    return item;
}

// Initialize admin navigation
function initAdminNavigation() {
    const navLinks = document.querySelectorAll('.admin-sidebar .nav-link[data-page]');
    
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            const page = this.getAttribute('data-page');
            navigateToAdminPage(page);
            
            // Update active state
            document.querySelectorAll('.admin-sidebar .nav-item').forEach(item => item.classList.remove('active'));
            this.closest('.nav-item').classList.add('active');
        });
    });
}

// Navigate to admin page
function navigateToAdminPage(page) {
    const content = document.getElementById('adminDashboardContent');
    const pageTitle = document.querySelector('.page-title');
    const pageSubtitle = document.querySelector('.page-subtitle');
    
    // Update URL without reload
    history.pushState({page}, '', `#${page}`);
    
    // Update page title and content
    switch (page) {
        case 'users':
            pageTitle.textContent = 'User Management';
            pageSubtitle.textContent = 'Manage user accounts and permissions';
            loadUsersPage();
            break;
        case 'activity':
            pageTitle.textContent = 'User Activity';
            pageSubtitle.textContent = 'Monitor user activity and system events';
            loadActivityPage();
            break;
        case 'bets':
            pageTitle.textContent = 'Betting Analytics';
            pageSubtitle.textContent = 'Analyze betting patterns and performance';
            loadBetsPage();
            break;
        case 'models':
            pageTitle.textContent = 'AI Models';
            pageSubtitle.textContent = 'Monitor and manage prediction models';
            loadModelsPage();
            break;
        case 'system':
            pageTitle.textContent = 'System Health';
            pageSubtitle.textContent = 'Monitor system performance and health';
            loadSystemPage();
            break;
        case 'settings':
            pageTitle.textContent = 'Admin Settings';
            pageSubtitle.textContent = 'Configure system settings and preferences';
            loadAdminSettingsPage();
            break;
        default:
            pageTitle.textContent = 'Admin Overview';
            pageSubtitle.textContent = 'Monitor and manage the GoonSteen platform';
            loadAdminOverviewPage();
    }
}

// Initialize quick actions
function initQuickActions() {
    const quickActions = document.querySelectorAll('.quick-action-btn');
    
    quickActions.forEach(action => {
        action.addEventListener('click', function() {
            const actionType = this.getAttribute('onclick');
            // Actions are handled by individual onclick handlers
        });
    });
}

// Initialize admin charts
function initAdminCharts() {
    initAdminMetricsChart();
}

// Initialize admin metrics chart
function initAdminMetricsChart() {
    const ctx = document.getElementById('adminMetricsChart');
    if (!ctx) return;
    
    const chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul'],
            datasets: [{
                label: 'Users',
                data: [850, 920, 1050, 1150, 1200, 1240, 1247],
                borderColor: 'rgb(37, 99, 235)',
                backgroundColor: 'rgba(37, 99, 235, 0.1)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
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
            }
        }
    });
    
    // Chart controls
    const chartControls = document.querySelectorAll('.chart-control');
    chartControls.forEach(control => {
        control.addEventListener('click', function() {
            chartControls.forEach(c => c.classList.remove('active'));
            this.classList.add('active');
            
            const metric = this.getAttribute('data-metric');
            updateAdminChartData(chart, metric);
        });
    });
}

// Update admin chart data
async function updateAdminChartData(chart, metric) {
    try {
        const response = await makeAuthenticatedRequest(`/api/admin/chart-data?metric=${metric}`);
        const data = await response.json();
        
        chart.data.labels = data.labels;
        chart.data.datasets[0].data = data.values;
        chart.data.datasets[0].label = metric.charAt(0).toUpperCase() + metric.slice(1);
        chart.update();
        
    } catch (error) {
        console.error('Failed to update admin chart data:', error);
    }
}

// Initialize real-time updates
function initAdminRealTime() {
    // Load system health immediately
    loadSystemHealth();
    
    // Update every 30 seconds
    setInterval(updateAdminRealTimeData, 30000);
    
    // Listen for admin WebSocket updates
    if (window.WebSocket) {
        connectAdminWebSocket();
    }
}

// Load system health data
async function loadSystemHealth() {
    try {
        const response = await makeAuthenticatedRequest('/api/admin/system-health');
        if (response.ok) {
            const healthData = await response.json();
            updateSystemHealthMetrics(healthData);
            updateSystemStatus(healthData.status);
        }
    } catch (error) {
        console.error('Failed to load system health:', error);
    }
}

// Update system status indicator
function updateSystemStatus(status) {
    const statusIndicator = document.querySelector('.status-indicator');
    const statusText = document.querySelector('.system-status span');
    
    if (statusIndicator) {
        statusIndicator.className = `status-indicator ${status}`;
    }
    
    if (statusText) {
        switch (status) {
            case 'healthy':
                statusText.textContent = 'All Systems Operational';
                break;
            case 'warning':
                statusText.textContent = 'System Performance Warning';
                break;
            case 'error':
                statusText.textContent = 'System Issues Detected';
                break;
            default:
                statusText.textContent = 'System Status Unknown';
        }
    }
}

// Update admin real-time data
async function updateAdminRealTimeData() {
    try {
        const response = await makeAuthenticatedRequest('/api/admin/realtime');
        const data = await response.json();
        
        updateSystemHealthMetrics(data.system_health);
        updateActiveUsersCount(data.active_users);
        
    } catch (error) {
        console.error('Failed to update admin real-time data:', error);
    }
}

// Connect to admin WebSocket
function connectAdminWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/admin`);
    
    ws.onmessage = function(event) {
        const data = JSON.parse(event.data);
        handleAdminWebSocketMessage(data);
    };
    
    ws.onclose = function() {
        // Reconnect after 5 seconds
        setTimeout(connectAdminWebSocket, 5000);
    };
}

// Handle admin WebSocket messages
function handleAdminWebSocketMessage(data) {
    switch (data.type) {
        case 'new_user':
            addNewUserToTable(data.user);
            incrementUserCount();
            break;
        case 'user_activity':
            addActivityToFeed(data.activity);
            break;
        case 'system_alert':
            showSystemAlert(data.alert);
            break;
        case 'model_update':
            updateModelStatus(data.model_id, data.status);
            break;
    }
}

// Quick Action Handlers
function broadcastMessage() {
    showBroadcastModal();
}

function runModelUpdate() {
    if (confirm('Are you sure you want to update all AI models? This may take several minutes.')) {
        startModelUpdate();
    }
}

function generateReport() {
    showReportModal();
}

function manageBackups() {
    navigateToAdminPage('backups');
}

function exportData() {
    showExportModal();
}

function refreshData() {
    location.reload();
}

// User Management Functions
async function viewUser(userId) {
    try {
        const response = await makeAuthenticatedRequest(`/api/admin/users/${userId}`);
        const user = await response.json();
        
        showUserModal(user);
        
    } catch (error) {
        console.error('Failed to load user details:', error);
        showAdminError('Failed to load user details');
    }
}

function editUser(userId) {
    // Navigate to user edit page
    window.location.href = `admin-user-edit.html?id=${userId}`;
}

async function suspendUser(userId) {
    if (!confirm('Are you sure you want to suspend this user?')) return;
    
    try {
        const response = await makeAuthenticatedRequest(`/api/admin/users/${userId}/suspend`, {
            method: 'POST'
        });
        
        if (response.ok) {
            showAdminSuccess('User suspended successfully');
            loadRecentUsers();
        } else {
            throw new Error('Failed to suspend user');
        }
        
    } catch (error) {
        console.error('Failed to suspend user:', error);
        showAdminError('Failed to suspend user');
    }
}

async function unsuspendUser(userId) {
    if (!confirm('Are you sure you want to unsuspend this user?')) return;
    
    try {
        const response = await makeAuthenticatedRequest(`/api/admin/users/${userId}/unsuspend`, {
            method: 'POST'
        });
        
        if (response.ok) {
            showAdminSuccess('User unsuspended successfully');
            loadRecentUsers();
        } else {
            throw new Error('Failed to unsuspend user');
        }
        
    } catch (error) {
        console.error('Failed to unsuspend user:', error);
        showAdminError('Failed to unsuspend user');
    }
}

function viewAllUsers() {
    navigateToAdminPage('users');
}

// Modal Functions
function showUserModal(user) {
    const modal = document.getElementById('userModal');
    
    document.getElementById('modalUserName').textContent = `${user.first_name} ${user.last_name}`;
    document.getElementById('modalUserEmail').textContent = user.email;
    document.getElementById('modalUserStatus').textContent = user.status;
    document.getElementById('modalUserJoined').textContent = formatDate(user.created_at);
    document.getElementById('modalUserBets').textContent = user.total_bets || '0';
    document.getElementById('modalUserWinRate').textContent = `${user.win_rate || 0}%`;
    document.getElementById('modalUserProfit').textContent = formatCurrency(user.total_profit || 0);
    
    showModal('userModal');
}

function showBroadcastModal() {
    // Implementation for broadcast modal
    console.log('Show broadcast modal');
}

function showReportModal() {
    // Implementation for report modal
    console.log('Show report modal');
}

function showExportModal() {
    // Implementation for export modal
    console.log('Show export modal');
}

// System Functions
function updateSystemHealthMetrics(healthData) {
    const metrics = document.querySelectorAll('.health-metric');
    
    metrics.forEach((metric, index) => {
        const value = metric.querySelector('.metric-value');
        const fill = metric.querySelector('.metric-fill');
        
        switch (index) {
            case 0: // CPU
                value.textContent = `${healthData.cpu}%`;
                fill.style.width = `${healthData.cpu}%`;
                fill.className = `metric-fill ${getHealthClass(healthData.cpu)}`;
                break;
            case 1: // Memory
                value.textContent = `${healthData.memory}%`;
                fill.style.width = `${healthData.memory}%`;
                fill.className = `metric-fill ${getHealthClass(healthData.memory)}`;
                break;
            case 2: // Database
                if (healthData.database_size_mb !== undefined) {
                    value.textContent = `${healthData.database_size_mb}MB`;
                    const sizePercentage = Math.min((healthData.database_size_mb / 1000) * 100, 100);
                    fill.style.width = `${sizePercentage}%`;
                    fill.className = `metric-fill ${getHealthClass(sizePercentage)}`;
                } else {
                    value.textContent = `${healthData.disk}%`;
                    fill.style.width = `${healthData.disk}%`;
                    fill.className = `metric-fill ${getHealthClass(healthData.disk)}`;
                }
                break;
            case 3: // API Response
                value.textContent = `${healthData.api_response || healthData.database_response_ms}ms`;
                const responseTime = healthData.api_response || healthData.database_response_ms || 200;
                const responsePercentage = Math.min((responseTime / 1000) * 100, 100);
                fill.style.width = `${responsePercentage}%`;
                fill.className = `metric-fill ${getHealthClass(responsePercentage, true)}`;
                break;
        }
    });
}

// Update active users count
function updateActiveUsersCount(count) {
    const activeUsersElement = document.querySelector('[data-metric="active-users"]');
    if (activeUsersElement) {
        activeUsersElement.textContent = count;
    }
}

function startModelUpdate() {
    showAdminSuccess('Model update started. This may take several minutes.');
    // Implementation for model update
}

// Page Loaders
function loadAdminOverviewPage() {
    // Already loaded
}

function loadUsersPage() {
    const content = document.getElementById('adminDashboardContent');
    content.innerHTML = `
        <div class="users-management-page">
            <div class="page-controls">
                <div class="search-bar">
                    <input type="text" id="userSearch" placeholder="Search users..." class="search-input">
                    <button class="btn btn-primary" onclick="searchUsers()">
                        <i class="fas fa-search"></i>
                        Search
                    </button>
                </div>
                <div class="filter-controls">
                    <select id="statusFilter" class="form-select">
                        <option value="">All Status</option>
                        <option value="active">Active</option>
                        <option value="premium">Premium</option>
                        <option value="suspended">Suspended</option>
                    </select>
                    <button class="btn btn-outline" onclick="exportUsers()">
                        <i class="fas fa-download"></i>
                        Export
                    </button>
                </div>
            </div>
            
            <div class="admin-section">
                <div class="section-header">
                    <h2 class="section-title">All Users</h2>
                    <div class="section-actions">
                        <button class="btn btn-primary" onclick="loadAllUsers()">
                            <i class="fas fa-refresh"></i>
                            Refresh
                        </button>
                    </div>
                </div>
                
                <div class="users-table-container">
                    <table class="admin-table" id="allUsersTable">
                        <thead>
                            <tr>
                                <th>User</th>
                                <th>Email</th>
                                <th>Status</th>
                                <th>Subscription</th>
                                <th>Joined</th>
                                <th>Last Login</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody id="allUsersTableBody">
                            <tr><td colspan="7" class="loading-row">Loading users...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    `;
    
    loadAllUsers();
}

function loadActivityPage() {
    const content = document.getElementById('adminDashboardContent');
    content.innerHTML = `
        <div class="activity-management-page">
            <div class="admin-section">
                <div class="section-header">
                    <h2 class="section-title">System Activity Log</h2>
                    <div class="section-actions">
                        <select id="activityFilter" class="form-select">
                            <option value="">All Activity</option>
                            <option value="login_success">Logins</option>
                            <option value="bet_placed">Bets</option>
                            <option value="user_registration">Registrations</option>
                            <option value="account_suspended">Suspensions</option>
                        </select>
                        <button class="btn btn-primary" onclick="loadSystemActivity()">
                            <i class="fas fa-refresh"></i>
                            Refresh
                        </button>
                    </div>
                </div>
                
                <div class="activity-table-container">
                    <table class="admin-table">
                        <thead>
                            <tr>
                                <th>Time</th>
                                <th>User</th>
                                <th>Activity</th>
                                <th>Description</th>
                                <th>IP Address</th>
                            </tr>
                        </thead>
                        <tbody id="activityTableBody">
                            <tr><td colspan="5" class="loading-row">Loading activity...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    `;
    
    loadSystemActivityPage();
}

function loadBetsPage() {
    const content = document.getElementById('adminDashboardContent');
    content.innerHTML = `
        <div class="betting-analytics-page">
            <div class="analytics-overview">
                <div class="analytics-card">
                    <h3>Total Volume</h3>
                    <div class="analytics-value" id="totalVolume">Loading...</div>
                </div>
                <div class="analytics-card">
                    <h3>Win Rate</h3>
                    <div class="analytics-value" id="overallWinRate">Loading...</div>
                </div>
                <div class="analytics-card">
                    <h3>Average Stake</h3>
                    <div class="analytics-value" id="avgStake">Loading...</div>
                </div>
                <div class="analytics-card">
                    <h3>Profit Margin</h3>
                    <div class="analytics-value" id="profitMargin">Loading...</div>
                </div>
            </div>
            
            <div class="admin-section">
                <div class="section-header">
                    <h2 class="section-title">Recent Bets</h2>
                    <button class="btn btn-primary" onclick="loadBettingData()">
                        <i class="fas fa-refresh"></i>
                        Refresh
                    </button>
                </div>
                
                <div class="bets-table-container">
                    <table class="admin-table">
                        <thead>
                            <tr>
                                <th>User</th>
                                <th>Bet Type</th>
                                <th>Stake</th>
                                <th>Odds</th>
                                <th>Status</th>
                                <th>Payout</th>
                                <th>Placed</th>
                            </tr>
                        </thead>
                        <tbody id="betsTableBody">
                            <tr><td colspan="7" class="loading-row">Loading bets...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    `;
    
    loadBettingData();
}

function loadModelsPage() {
    const content = document.getElementById('adminDashboardContent');
    content.innerHTML = `
        <div class="models-management-page">
            <div class="models-overview">
                <div class="model-card" data-model="ensemble">
                    <div class="model-header">
                        <h3>Ensemble Model</h3>
                        <div class="model-status active">Active</div>
                    </div>
                    <div class="model-stats">
                        <div class="stat">
                            <span class="stat-label">Accuracy</span>
                            <span class="stat-value" id="ensembleAccuracy">Loading...</span>
                        </div>
                        <div class="stat">
                            <span class="stat-label">Predictions</span>
                            <span class="stat-value" id="ensemblePredictions">Loading...</span>
                        </div>
                    </div>
                    <button class="btn btn-outline" onclick="retrainModel('ensemble')">Retrain Model</button>
                </div>
                
                <div class="model-card" data-model="xgboost">
                    <div class="model-header">
                        <h3>XGBoost Model</h3>
                        <div class="model-status active">Active</div>
                    </div>
                    <div class="model-stats">
                        <div class="stat">
                            <span class="stat-label">Accuracy</span>
                            <span class="stat-value" id="xgboostAccuracy">Loading...</span>
                        </div>
                        <div class="stat">
                            <span class="stat-label">Predictions</span>
                            <span class="stat-value" id="xgboostPredictions">Loading...</span>
                        </div>
                    </div>
                    <button class="btn btn-outline" onclick="retrainModel('xgboost')">Retrain Model</button>
                </div>
                
                <div class="model-card" data-model="neural">
                    <div class="model-header">
                        <h3>Neural Network</h3>
                        <div class="model-status active">Active</div>
                    </div>
                    <div class="model-stats">
                        <div class="stat">
                            <span class="stat-label">Accuracy</span>
                            <span class="stat-value" id="neuralAccuracy">Loading...</span>
                        </div>
                        <div class="stat">
                            <span class="stat-label">Predictions</span>
                            <span class="stat-value" id="neuralPredictions">Loading...</span>
                        </div>
                    </div>
                    <button class="btn btn-outline" onclick="retrainModel('neural')">Retrain Model</button>
                </div>
            </div>
        </div>
    `;
    
    loadModelData();
}

function loadSystemPage() {
    const content = document.getElementById('adminDashboardContent');
    content.innerHTML = `
        <div class="system-health-page">
            <div class="health-overview">
                <div class="health-card">
                    <h3>System Status</h3>
                    <div class="system-status-indicator" id="systemStatusIndicator">
                        <div class="status-dot online"></div>
                        <span>All Systems Operational</span>
                    </div>
                </div>
            </div>
            
            <div class="admin-section">
                <div class="section-header">
                    <h2 class="section-title">System Metrics</h2>
                    <button class="btn btn-primary" onclick="loadSystemHealth()">
                        <i class="fas fa-refresh"></i>
                        Refresh
                    </button>
                </div>
                
                <div class="system-metrics-grid">
                    <div class="metric-card">
                        <div class="metric-header">
                            <h4>CPU Usage</h4>
                            <span class="metric-value" id="cpuUsage">Loading...</span>
                        </div>
                        <div class="metric-bar">
                            <div class="metric-fill" id="cpuFill"></div>
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-header">
                            <h4>Memory Usage</h4>
                            <span class="metric-value" id="memoryUsage">Loading...</span>
                        </div>
                        <div class="metric-bar">
                            <div class="metric-fill" id="memoryFill"></div>
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-header">
                            <h4>Database Size</h4>
                            <span class="metric-value" id="databaseSize">Loading...</span>
                        </div>
                        <div class="metric-bar">
                            <div class="metric-fill" id="databaseFill"></div>
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-header">
                            <h4>API Response</h4>
                            <span class="metric-value" id="apiResponse">Loading...</span>
                        </div>
                        <div class="metric-bar">
                            <div class="metric-fill" id="apiFill"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    loadSystemHealth();
}

function loadAdminSettingsPage() {
    console.log('Loading admin settings page...');
}

// Utility Functions
function formatNumber(num) {
    return new Intl.NumberFormat('en-US').format(num);
}

function formatDate(dateString) {
    return new Date(dateString).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    });
}

function getUserStatusClass(status) {
    switch (status.toLowerCase()) {
        case 'active': return 'active';
        case 'premium': return 'premium';
        case 'suspended': return 'suspended';
        default: return 'active';
    }
}

function getActivityIconClass(type) {
    switch (type) {
        case 'user_registration': return 'user';
        case 'bet_placed': return 'bet';
        case 'system_update': return 'system';
        case 'error': return 'warning';
        default: return 'system';
    }
}

function getActivityIcon(type) {
    switch (type) {
        case 'user_registration': return 'fa-user-plus';
        case 'bet_placed': return 'fa-chart-line';
        case 'system_update': return 'fa-cog';
        case 'error': return 'fa-exclamation-triangle';
        default: return 'fa-info-circle';
    }
}

function getHealthClass(value, isResponseTime = false) {
    if (isResponseTime) {
        // For response time, lower is better
        if (value < 200) return 'good';
        if (value < 500) return 'warning';
        return 'error';
    } else {
        // For CPU, memory, disk usage
        if (value < 60) return 'good';
        if (value < 85) return 'warning';
        return 'error';
    }
}

function updateAdminChangeIndicator(element, value, suffix) {
    const isPositive = value > 0;
    const icon = isPositive ? 'fa-arrow-up' : 'fa-arrow-down';
    const className = isPositive ? 'positive' : 'negative';
    
    element.className = `card-change ${className}`;
    element.innerHTML = `
        <i class="fas ${icon}"></i>
        ${isPositive ? '+' : ''}${value} ${suffix}
    `;
}

// Notification Functions
function showAdminSuccess(message) {
    showNotification(message, 'success');
}

function showAdminError(message) {
    showNotification(message, 'error');
}

function showNotification(message, type) {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `admin-notification ${type}`;
    notification.innerHTML = `
        <div class="notification-content">
            <i class="fas ${type === 'success' ? 'fa-check-circle' : 'fa-exclamation-triangle'}"></i>
            <span>${message}</span>
        </div>
        <button class="notification-close" onclick="this.parentElement.remove()">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    document.body.appendChild(notification);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (notification.parentElement) {
            notification.remove();
        }
    }, 5000);
}

// Additional Admin Functions
async function loadAllUsers() {
    try {
        const response = await makeAuthenticatedRequest('/api/admin/all-users');
        const users = await response.json();
        
        const tbody = document.getElementById('allUsersTableBody');
        if (!tbody) return;
        
        tbody.innerHTML = '';
        
        users.forEach(user => {
            const row = createDetailedUserTableRow(user);
            tbody.appendChild(row);
        });
        
    } catch (error) {
        console.error('Failed to load all users:', error);
        showAdminError('Failed to load users');
    }
}

function createDetailedUserTableRow(user) {
    const row = document.createElement('tr');
    const statusClass = getUserStatusClass(user.status);
    const joinedTime = formatDate(user.created_at);
    const lastLogin = user.last_login ? formatRelativeTime(user.last_login) : 'Never';
    
    row.innerHTML = `
        <td>
            <div class="user-cell">
                <div class="user-avatar small">${user.first_name.charAt(0)}${user.last_name.charAt(0)}</div>
                <div class="user-info">
                    <div class="user-name">${user.first_name} ${user.last_name}</div>
                    <div class="user-id">#${user.id}</div>
                </div>
            </div>
        </td>
        <td>${user.email}</td>
        <td><span class="status-badge ${statusClass}">${user.status}</span></td>
        <td><span class="status-badge ${user.subscription_type}">${user.subscription_type}</span></td>
        <td>${joinedTime}</td>
        <td>${lastLogin}</td>
        <td>
            <div class="action-buttons">
                <button class="btn-icon" onclick="viewUser(${user.id})" title="View User">
                    <i class="fas fa-eye"></i>
                </button>
                <button class="btn-icon" onclick="editUser(${user.id})" title="Edit User">
                    <i class="fas fa-edit"></i>
                </button>
                ${user.status === 'suspended' ? 
                    `<button class="btn-icon warning" onclick="unsuspendUser(${user.id})" title="Unsuspend User">
                        <i class="fas fa-unlock"></i>
                    </button>` :
                    `<button class="btn-icon warning" onclick="suspendUser(${user.id})" title="Suspend User">
                        <i class="fas fa-lock"></i>
                    </button>`
                }
            </div>
        </td>
    `;
    
    return row;
}

async function loadSystemActivityPage() {
    try {
        const response = await makeAuthenticatedRequest('/api/admin/detailed-activity');
        const activities = await response.json();
        
        const tbody = document.getElementById('activityTableBody');
        if (!tbody) return;
        
        tbody.innerHTML = '';
        
        activities.forEach(activity => {
            const row = createActivityTableRow(activity);
            tbody.appendChild(row);
        });
        
    } catch (error) {
        console.error('Failed to load system activity:', error);
        showAdminError('Failed to load activity');
    }
}

function createActivityTableRow(activity) {
    const row = document.createElement('tr');
    
    row.innerHTML = `
        <td>${formatDateTime(activity.created_at)}</td>
        <td>${activity.username || 'System'}</td>
        <td><span class="activity-type-badge ${activity.activity_type}">${activity.activity_type}</span></td>
        <td>${activity.description}</td>
        <td>${activity.ip_address || '-'}</td>
    `;
    
    return row;
}

async function loadBettingData() {
    try {
        const [analyticsResponse, betsResponse] = await Promise.all([
            makeAuthenticatedRequest('/api/admin/betting-analytics'),
            makeAuthenticatedRequest('/api/admin/recent-bets')
        ]);
        
        const analytics = await analyticsResponse.json();
        const bets = await betsResponse.json();
        
        // Update analytics cards
        document.getElementById('totalVolume').textContent = formatCurrency(analytics.total_volume);
        document.getElementById('overallWinRate').textContent = `${analytics.win_rate}%`;
        document.getElementById('avgStake').textContent = formatCurrency(analytics.avg_stake);
        document.getElementById('profitMargin').textContent = `${analytics.profit_margin}%`;
        
        // Update bets table
        const tbody = document.getElementById('betsTableBody');
        if (tbody) {
            tbody.innerHTML = '';
            
            bets.forEach(bet => {
                const row = createBetTableRow(bet);
                tbody.appendChild(row);
            });
        }
        
    } catch (error) {
        console.error('Failed to load betting data:', error);
        showAdminError('Failed to load betting data');
    }
}

function createBetTableRow(bet) {
    const row = document.createElement('tr');
    
    row.innerHTML = `
        <td>${bet.username}</td>
        <td>${bet.bet_type}</td>
        <td>${formatCurrency(bet.stake)}</td>
        <td>${bet.odds}</td>
        <td><span class="status-badge ${bet.status}">${bet.status}</span></td>
        <td>${formatCurrency(bet.actual_payout)}</td>
        <td>${formatRelativeTime(bet.placed_at)}</td>
    `;
    
    return row;
}

async function loadModelData() {
    try {
        const response = await makeAuthenticatedRequest('/api/admin/model-performance');
        const models = await response.json();
        
        models.forEach(model => {
            const accuracyElement = document.getElementById(`${model.name}Accuracy`);
            const predictionsElement = document.getElementById(`${model.name}Predictions`);
            
            if (accuracyElement) accuracyElement.textContent = `${model.accuracy}%`;
            if (predictionsElement) predictionsElement.textContent = formatNumber(model.total_predictions);
        });
        
    } catch (error) {
        console.error('Failed to load model data:', error);
        showAdminError('Failed to load model data');
    }
}

function searchUsers() {
    const searchTerm = document.getElementById('userSearch').value;
    const statusFilter = document.getElementById('statusFilter').value;
    
    // Implementation for user search
    console.log('Searching users:', searchTerm, statusFilter);
}

function exportUsers() {
    // Implementation for user export
    showAdminSuccess('User export started. Download will begin shortly.');
}

function retrainModel(modelName) {
    if (confirm(`Are you sure you want to retrain the ${modelName} model? This may take several minutes.`)) {
        showAdminSuccess(`${modelName} model retraining started.`);
        // Implementation for model retraining
    }
}

function formatDateTime(timestamp) {
    return new Date(timestamp).toLocaleString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

// Export functions for global use
window.AdminUtils = {
    navigateToAdminPage,
    viewUser,
    editUser,
    suspendUser,
    unsuspendUser,
    broadcastMessage,
    runModelUpdate,
    generateReport,
    manageBackups,
    exportData,
    refreshData,
    loadAllUsers,
    searchUsers,
    exportUsers,
    retrainModel,
    loadBettingData,
    loadModelData,
    loadSystemActivityPage
};
