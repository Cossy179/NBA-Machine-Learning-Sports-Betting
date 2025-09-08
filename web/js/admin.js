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
        const response = await makeAuthenticatedRequest('/api/admin/overview');
        const data = await response.json();
        
        updateAdminOverviewCards(data);
        
    } catch (error) {
        console.error('Failed to load admin overview:', error);
        showAdminError('Failed to load overview data');
    }
}

// Update admin overview cards
function updateAdminOverviewCards(data) {
    const cards = document.querySelectorAll('.admin-card');
    
    cards.forEach((card, index) => {
        const value = card.querySelector('.card-value');
        const change = card.querySelector('.card-change');
        const progressFill = card.querySelector('.progress-fill');
        const progressText = card.querySelector('.card-progress span');
        
        switch (index) {
            case 0: // Total Users
                value.textContent = formatNumber(data.total_users);
                updateAdminChangeIndicator(change, data.new_users_week, 'this week');
                if (progressFill) progressFill.style.width = `${data.active_users_percentage}%`;
                if (progressText) progressText.textContent = `${data.active_users_percentage}% active`;
                break;
                
            case 1: // Total Bets
                value.textContent = formatNumber(data.total_bets);
                updateAdminChangeIndicator(change, data.bets_today, 'today');
                if (progressFill) progressFill.style.width = `${data.win_rate}%`;
                if (progressText) progressText.textContent = `${data.win_rate}% win rate`;
                break;
                
            case 2: // Revenue
                value.textContent = formatCurrency(data.revenue);
                updateAdminChangeIndicator(change, data.revenue_growth, '% this month');
                if (progressFill) progressFill.style.width = `${data.revenue_target_percentage}%`;
                if (progressText) progressText.textContent = `${data.revenue_target_percentage}% of target`;
                break;
                
            case 3: // Model Accuracy
                value.textContent = `${data.model_accuracy}%`;
                updateAdminChangeIndicator(change, data.accuracy_improvement, '% improvement');
                if (progressFill) progressFill.style.width = `${data.model_accuracy}%`;
                if (progressText) progressText.textContent = 'Above benchmark';
                break;
        }
    });
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
    // Update every 30 seconds
    setInterval(updateAdminRealTimeData, 30000);
    
    // Listen for admin WebSocket updates
    if (window.WebSocket) {
        connectAdminWebSocket();
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
                value.textContent = `${healthData.database}%`;
                fill.style.width = `${healthData.database}%`;
                fill.className = `metric-fill ${getHealthClass(healthData.database)}`;
                break;
            case 3: // API Response
                value.textContent = `${healthData.api_response}ms`;
                const responsePercentage = Math.min((healthData.api_response / 1000) * 100, 100);
                fill.style.width = `${responsePercentage}%`;
                fill.className = `metric-fill ${getHealthClass(responsePercentage)}`;
                break;
        }
    });
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
    console.log('Loading users page...');
}

function loadActivityPage() {
    console.log('Loading activity page...');
}

function loadBetsPage() {
    console.log('Loading bets page...');
}

function loadModelsPage() {
    console.log('Loading models page...');
}

function loadSystemPage() {
    console.log('Loading system page...');
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

function getHealthClass(value) {
    if (value < 50) return 'good';
    if (value < 80) return 'warning';
    return 'error';
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
    
    // Add styles if not already added
    if (!document.querySelector('#admin-notification-styles')) {
        const style = document.createElement('style');
        style.id = 'admin-notification-styles';
        style.textContent = `
            .admin-notification {
                position: fixed;
                top: 2rem;
                right: 2rem;
                background: white;
                border-radius: 0.75rem;
                padding: 1rem 1.5rem;
                box-shadow: var(--shadow-xl);
                border-left: 4px solid var(--accent-color);
                z-index: 10000;
                display: flex;
                align-items: center;
                gap: 1rem;
                min-width: 300px;
                animation: slideInRight 0.3s ease;
            }
            
            .admin-notification.error {
                border-left-color: #ef4444;
            }
            
            .admin-notification.success {
                border-left-color: var(--accent-color);
            }
            
            .notification-content {
                display: flex;
                align-items: center;
                gap: 0.75rem;
                flex: 1;
            }
            
            .notification-content i {
                color: var(--accent-color);
                font-size: 1.125rem;
            }
            
            .admin-notification.error .notification-content i {
                color: #ef4444;
            }
            
            .notification-close {
                background: none;
                border: none;
                color: var(--text-light);
                cursor: pointer;
                padding: 0.25rem;
                border-radius: 0.25rem;
                transition: all 0.2s ease;
            }
            
            .notification-close:hover {
                background: var(--bg-secondary);
                color: var(--text-secondary);
            }
            
            @keyframes slideInRight {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
        `;
        document.head.appendChild(style);
    }
    
    document.body.appendChild(notification);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (notification.parentElement) {
            notification.remove();
        }
    }, 5000);
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
    refreshData
};
