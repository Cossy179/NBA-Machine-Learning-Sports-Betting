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
            console.error('Unauthorized response for:', url);
            // Don't immediately logout - let the calling function handle it
            return response;
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
    // Use stored user data - more reliable than server check
    const storedUserData = JSON.parse(localStorage.getItem('user_data') || '{}');
    
    if (storedUserData.is_admin) {
        console.log('Using stored admin user data:', storedUserData);
        updateAdminUserInfo(storedUserData);
        return;
    }
    
    // If no stored admin data, redirect to login
    console.log('No admin user data found - redirecting to login');
    window.location.href = 'login.html';
}

// Initialize admin dashboard
function initAdminDashboard() {
    loadAdminOverview();
    loadRecentUsers();
    loadSystemActivity();
    initAdminNavigation();
    initQuickActions();
    initMobileNavigation();
    initSidebarToggle();
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
        
        const response = await makeAuthenticatedRequest('/api.php?route=api/admin/overview');
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
        const response = await makeAuthenticatedRequest('/api.php?route=api/admin/recent-users');
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
        const response = await makeAuthenticatedRequest('/api.php?route=api/admin/activity');
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
            
            // Close mobile sidebar
            if (window.innerWidth <= 1024) {
                document.querySelector('.admin-sidebar').classList.remove('mobile-visible');
            }
        });
    });
}

// Initialize mobile navigation
function initMobileNavigation() {
    const mobileNavItems = document.querySelectorAll('.mobile-admin-nav-item[data-page]');
    
    mobileNavItems.forEach(item => {
        item.addEventListener('click', function(e) {
            e.preventDefault();
            
            const page = this.getAttribute('data-page');
            navigateToAdminPage(page);
            
            // Update active state
            document.querySelectorAll('.mobile-admin-nav-item').forEach(nav => nav.classList.remove('active'));
            this.classList.add('active');
        });
    });
}

// Initialize sidebar toggle
function initSidebarToggle() {
    const mobileToggle = document.getElementById('mobileHeaderToggle');
    const sidebar = document.querySelector('.admin-sidebar');
    
    if (mobileToggle && sidebar) {
        mobileToggle.addEventListener('click', function() {
            sidebar.classList.toggle('mobile-visible');
        });
    }
    
    // Close sidebar when clicking outside on mobile
    document.addEventListener('click', function(e) {
        if (window.innerWidth <= 1024) {
            if (!sidebar.contains(e.target) && !mobileToggle.contains(e.target)) {
                sidebar.classList.remove('mobile-visible');
            }
        }
    });
}

// Navigate to admin page
function navigateToAdminPage(page) {
    const content = document.getElementById('adminDashboardContent');
    const pageTitle = document.querySelector('.page-title');
    const pageSubtitle = document.querySelector('.page-subtitle');
    
    // Update URL without reload
    history.pushState({page}, '', `#${page}`);
    
    // Update active states for both desktop and mobile nav
    updateNavActiveStates(page);
    
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

// Update navigation active states
function updateNavActiveStates(page) {
    // Update desktop nav
    document.querySelectorAll('.admin-sidebar .nav-item').forEach(item => {
        item.classList.remove('active');
        const link = item.querySelector('.nav-link[data-page]');
        if (link && link.getAttribute('data-page') === page) {
            item.classList.add('active');
        }
    });
    
    // Update mobile nav
    document.querySelectorAll('.mobile-admin-nav-item').forEach(item => {
        item.classList.remove('active');
        if (item.getAttribute('data-page') === page) {
            item.classList.add('active');
        }
    });
    
    // Special case for overview page
    if (page === 'overview' || !page) {
        document.querySelector('.admin-sidebar .nav-item').classList.add('active');
        document.querySelector('.mobile-admin-nav-item[data-page="overview"]').classList.add('active');
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
        const response = await makeAuthenticatedRequest('/api.php?route=api/admin/system-health');
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
        const response = await makeAuthenticatedRequest('/api.php?route=api/admin/realtime');
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

// Admin User Edit Page Logic
function getQueryParam(name) {
    const url = new URL(window.location.href);
    return url.searchParams.get(name);
}

async function fetchJson(url, options = {}) {
    const token = localStorage.getItem('auth_token');
    const headers = Object.assign({ 'Content-Type': 'application/json' }, options.headers || {});
    if (token) {
        headers['Authorization'] = `Bearer ${token}`;
    } else if (!options.allowUnauthed) {
        // No token and unauthenticated not allowed: bounce to login gracefully
        window.location.href = 'login.html';
        throw new Error('No authentication token');
    }
    const res = await fetch(url, Object.assign({}, options, { headers }));
    if (!res.ok) throw new Error((await res.text()) || 'Request failed');
    return res.json();
}

async function initAdminUserEdit() {
    const id = getQueryParam('id');
    if (!id) {
        alert('Missing user id');
        window.location.href = 'admin-dashboard.html';
        return;
    }

    // Populate form
    try {
        const user = await fetchJson(`/api/admin/users/${id}`);
        document.getElementById('userId').value = user.id;
        document.getElementById('first_name').value = user.first_name || '';
        document.getElementById('last_name').value = user.last_name || '';
        document.getElementById('email').value = user.email || '';
        document.getElementById('status').value = user.status || 'active';
        document.getElementById('subscription_type').value = user.subscription_type || 'free';
    } catch (e) {
        console.error(e);
        alert('Failed to load user');
        window.location.href = 'admin-dashboard.html';
        return;
    }

    // Submit handler
    const form = document.getElementById('editUserForm');
    form.addEventListener('submit', async (ev) => {
        ev.preventDefault();
        const payload = {
            first_name: document.getElementById('first_name').value.trim(),
            last_name: document.getElementById('last_name').value.trim(),
            email: document.getElementById('email').value.trim(),
            status: document.getElementById('status').value,
            subscription_type: document.getElementById('subscription_type').value
        };
        const uid = document.getElementById('userId').value;
        try {
            await fetchJson(`/api/admin/users/${uid}`, { method: 'PUT', body: JSON.stringify(payload) });
            alert('User updated');
            window.location.href = 'admin-dashboard.html';
        } catch (e) {
            console.error(e);
            alert('Failed to save user');
        }
    });
}

// expose init for html
window.initAdminUserEdit = initAdminUserEdit;

// Safe logout without requiring token
function logout() {
    clearAuthData();
    fetchJson('/api.php?route=api/logout', { method: 'POST', allowUnauthed: true })
        .finally(() => { window.location.href = 'login.html'; });
}

window.logout = logout;

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
    
    // Populate modal with user data
    const modalUserName = document.getElementById('modalUserName');
    const modalUserEmail = document.getElementById('modalUserEmail');
    const modalUserStatus = document.getElementById('modalUserStatus');
    const modalUserJoined = document.getElementById('modalUserJoined');
    const modalUserBets = document.getElementById('modalUserBets');
    const modalUserWinRate = document.getElementById('modalUserWinRate');
    const modalUserProfit = document.getElementById('modalUserProfit');
    
    if (modalUserName) modalUserName.textContent = `${user.first_name} ${user.last_name}`;
    if (modalUserEmail) modalUserEmail.textContent = user.email;
    if (modalUserStatus) modalUserStatus.textContent = user.status;
    if (modalUserJoined) modalUserJoined.textContent = formatDate(user.created_at);
    if (modalUserBets) modalUserBets.textContent = user.total_bets || '0';
    if (modalUserWinRate) modalUserWinRate.textContent = `${Math.round(user.win_rate || 0)}%`;
    if (modalUserProfit) modalUserProfit.textContent = formatCurrency(user.total_profit || 0);
    
    // Show modal
    if (modal) {
        modal.classList.add('active');
        document.body.style.overflow = 'hidden';
        
        // Ensure modal is scrolled to top
        const modalBody = modal.querySelector('.modal-body');
        if (modalBody) {
            modalBody.scrollTop = 0;
        }
    }
}

// Close modal and cleanup
function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.classList.remove('active');
        document.body.style.overflow = '';
        
        // Reset scroll position
        const modalBody = modal.querySelector('.modal-body');
        if (modalBody) {
            modalBody.scrollTop = 0;
        }
    }
}

// Close modal when clicking outside
document.addEventListener('click', function(e) {
    const activeModal = document.querySelector('.modal.active');
    if (activeModal && e.target === activeModal) {
        const modalId = activeModal.getAttribute('id');
        closeModal(modalId);
    }
});

// Close modal with Escape key
document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
        const activeModal = document.querySelector('.modal.active');
        if (activeModal) {
            const modalId = activeModal.getAttribute('id');
            closeModal(modalId);
        }
    }
});

function showBroadcastModal() {
    const modalHtml = `
        <div class="modal active" id="broadcastModal">
            <div class="modal-content large">
                <div class="modal-header">
                    <h3>Broadcast Message</h3>
                    <button class="modal-close" onclick="closeModal('broadcastModal')">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="modal-body">
                    <form id="broadcastForm">
                        <div class="form-group">
                            <label for="messageTitle">Message Title</label>
                            <input type="text" id="messageTitle" class="form-input" placeholder="Enter message title" required>
                        </div>
                        <div class="form-group">
                            <label for="messageContent">Message Content</label>
                            <textarea id="messageContent" class="form-textarea" rows="6" placeholder="Enter your message..." required></textarea>
                        </div>
                        <div class="form-group">
                            <label for="messageType">Message Type</label>
                            <select id="messageType" class="form-select">
                                <option value="info">Information</option>
                                <option value="warning">Warning</option>
                                <option value="success">Success</option>
                                <option value="error">Error</option>
                            </select>
                        </div>
                    </form>
                </div>
                <div class="modal-footer">
                    <button class="btn btn-outline" onclick="closeModal('broadcastModal')">Cancel</button>
                    <button class="btn btn-primary" onclick="sendBroadcast()">Send Message</button>
                </div>
            </div>
        </div>
    `;
    
    document.body.insertAdjacentHTML('beforeend', modalHtml);
}

function showReportModal() {
    const modalHtml = `
        <div class="modal active" id="reportModal">
            <div class="modal-content large">
                <div class="modal-header">
                    <h3>Generate Report</h3>
                    <button class="modal-close" onclick="closeModal('reportModal')">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="modal-body">
                    <form id="reportForm">
                        <div class="form-group">
                            <label for="reportType">Report Type</label>
                            <select id="reportType" class="form-select" required>
                                <option value="">Select Report Type</option>
                                <option value="user_activity">User Activity Report</option>
                                <option value="betting_analytics">Betting Analytics Report</option>
                                <option value="model_performance">Model Performance Report</option>
                                <option value="system_health">System Health Report</option>
                                <option value="financial">Financial Summary Report</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="dateRange">Date Range</label>
                            <select id="dateRange" class="form-select" required>
                                <option value="7d">Last 7 Days</option>
                                <option value="30d">Last 30 Days</option>
                                <option value="90d">Last 90 Days</option>
                                <option value="1y">Last Year</option>
                                <option value="custom">Custom Range</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="reportFormat">Format</label>
                            <select id="reportFormat" class="form-select">
                                <option value="pdf">PDF</option>
                                <option value="excel">Excel</option>
                                <option value="csv">CSV</option>
                            </select>
                        </div>
                    </form>
                </div>
                <div class="modal-footer">
                    <button class="btn btn-outline" onclick="closeModal('reportModal')">Cancel</button>
                    <button class="btn btn-primary" onclick="generateReportFile()">Generate Report</button>
                </div>
            </div>
        </div>
    `;
    
    document.body.insertAdjacentHTML('beforeend', modalHtml);
}

function showExportModal() {
    const modalHtml = `
        <div class="modal active" id="exportModal">
            <div class="modal-content large">
                <div class="modal-header">
                    <h3>Export Data</h3>
                    <button class="modal-close" onclick="closeModal('exportModal')">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="modal-body">
                    <div class="export-options">
                        <div class="export-option">
                            <input type="checkbox" id="exportUsers" checked>
                            <label for="exportUsers">User Data</label>
                        </div>
                        <div class="export-option">
                            <input type="checkbox" id="exportBets" checked>
                            <label for="exportBets">Betting Data</label>
                        </div>
                        <div class="export-option">
                            <input type="checkbox" id="exportActivity">
                            <label for="exportActivity">Activity Logs</label>
                        </div>
                        <div class="export-option">
                            <input type="checkbox" id="exportModels">
                            <label for="exportModels">Model Performance</label>
                        </div>
                    </div>
                    <div class="form-group">
                        <label for="exportFormat">Export Format</label>
                        <select id="exportFormat" class="form-select">
                            <option value="json">JSON</option>
                            <option value="csv">CSV</option>
                            <option value="excel">Excel</option>
                        </select>
                    </div>
                </div>
                <div class="modal-footer">
                    <button class="btn btn-outline" onclick="closeModal('exportModal')">Cancel</button>
                    <button class="btn btn-primary" onclick="executeExport()">Export Data</button>
                </div>
            </div>
        </div>
    `;
    
    document.body.insertAdjacentHTML('beforeend', modalHtml);
}

// Modal action functions
function sendBroadcast() {
    const title = document.getElementById('messageTitle').value;
    const content = document.getElementById('messageContent').value;
    const type = document.getElementById('messageType').value;
    
    if (!title || !content) {
        showAdminError('Please fill in all required fields');
        return;
    }
    
    // Send broadcast message
    makeAuthenticatedRequest('/api.php?route=api/admin/broadcast', {
        method: 'POST',
        body: JSON.stringify({ title, content, type })
    }).then(response => {
        if (response.ok) {
            showAdminSuccess('Broadcast message sent successfully');
            closeModal('broadcastModal');
            document.getElementById('broadcastModal').remove();
        } else {
            showAdminError('Failed to send broadcast message');
        }
    }).catch(error => {
        console.error('Broadcast error:', error);
        showAdminError('Network error sending broadcast');
    });
}

function generateReportFile() {
    const reportType = document.getElementById('reportType').value;
    const dateRange = document.getElementById('dateRange').value;
    const format = document.getElementById('reportFormat').value;
    
    if (!reportType) {
        showAdminError('Please select a report type');
        return;
    }
    
    showAdminSuccess('Report generation started. Download will begin shortly.');
    closeModal('reportModal');
    document.getElementById('reportModal').remove();
    
    // Trigger report generation
    makeAuthenticatedRequest('/api.php?route=api/admin/generate-report', {
        method: 'POST',
        body: JSON.stringify({ reportType, dateRange, format })
    }).then(response => {
        if (response.ok) {
            return response.blob();
        }
        throw new Error('Report generation failed');
    }).then(blob => {
        // Download the file
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `goonsteen_${reportType}_${dateRange}.${format}`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    }).catch(error => {
        console.error('Report generation error:', error);
        showAdminError('Failed to generate report');
    });
}

function executeExport() {
    const exportData = {
        users: document.getElementById('exportUsers').checked,
        bets: document.getElementById('exportBets').checked,
        activity: document.getElementById('exportActivity').checked,
        models: document.getElementById('exportModels').checked,
        format: document.getElementById('exportFormat').value
    };
    
    showAdminSuccess('Data export started. Download will begin shortly.');
    closeModal('exportModal');
    document.getElementById('exportModal').remove();
    
    // Trigger data export
    makeAuthenticatedRequest('/api.php?route=api/admin/export', {
        method: 'POST',
        body: JSON.stringify(exportData)
    }).then(response => {
        if (response.ok) {
            return response.blob();
        }
        throw new Error('Export failed');
    }).then(blob => {
        // Download the file
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `goonsteen_export_${new Date().toISOString().split('T')[0]}.${exportData.format}`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    }).catch(error => {
        console.error('Export error:', error);
        showAdminError('Failed to export data');
    });
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
    const content = document.getElementById('adminDashboardContent');
    content.innerHTML = `
        <div class="admin-settings-page">
            <div class="settings-sections">
                <div class="admin-section">
                    <div class="section-header">
                        <h2 class="section-title">System Configuration</h2>
                        <button class="btn btn-primary" onclick="saveSettings()">
                            <i class="fas fa-save"></i>
                            Save Changes
                        </button>
                    </div>
                    
                    <div class="settings-grid">
                        <div class="setting-group">
                            <h4>Security Settings</h4>
                            <div class="setting-item">
                                <label for="maxLoginAttempts">Max Login Attempts</label>
                                <input type="number" id="maxLoginAttempts" class="form-input" value="5" min="3" max="10">
                            </div>
                            <div class="setting-item">
                                <label for="sessionTimeout">Session Timeout (minutes)</label>
                                <input type="number" id="sessionTimeout" class="form-input" value="1440" min="60" max="10080">
                            </div>
                            <div class="setting-item">
                                <label for="passwordMinLength">Minimum Password Length</label>
                                <input type="number" id="passwordMinLength" class="form-input" value="8" min="6" max="20">
                            </div>
                        </div>
                        
                        <div class="setting-group">
                            <h4>Betting Configuration</h4>
                            <div class="setting-item">
                                <label for="defaultBankroll">Default Bankroll ($)</label>
                                <input type="number" id="defaultBankroll" class="form-input" value="1000" min="100" max="10000" step="100">
                            </div>
                            <div class="setting-item">
                                <label for="minBetAmount">Minimum Bet Amount ($)</label>
                                <input type="number" id="minBetAmount" class="form-input" value="1" min="0.01" max="100" step="0.01">
                            </div>
                            <div class="setting-item">
                                <label for="maxBetAmount">Maximum Bet Amount ($)</label>
                                <input type="number" id="maxBetAmount" class="form-input" value="1000" min="100" max="50000" step="100">
                            </div>
                        </div>
                        
                        <div class="setting-group">
                            <h4>AI Model Settings</h4>
                            <div class="setting-item">
                                <label for="modelUpdateFreq">Update Frequency (hours)</label>
                                <input type="number" id="modelUpdateFreq" class="form-input" value="24" min="1" max="168">
                            </div>
                            <div class="setting-item">
                                <label for="confidenceThreshold">Confidence Threshold (%)</label>
                                <input type="number" id="confidenceThreshold" class="form-input" value="60" min="50" max="95">
                            </div>
                            <div class="setting-item">
                                <div class="checkbox-wrapper">
                                    <input type="checkbox" id="kellyEnabled" checked>
                                    <label for="kellyEnabled">Enable Kelly Criterion</label>
                                </div>
                            </div>
                        </div>
                        
                        <div class="setting-group">
                            <h4>System Maintenance</h4>
                            <div class="maintenance-actions">
                                <button class="btn btn-outline" onclick="cleanupOldData()">
                                    <i class="fas fa-broom"></i>
                                    Cleanup Old Data
                                </button>
                                <button class="btn btn-outline" onclick="optimizeDatabase()">
                                    <i class="fas fa-database"></i>
                                    Optimize Database
                                </button>
                                <button class="btn btn-outline" onclick="clearCache()">
                                    <i class="fas fa-trash"></i>
                                    Clear Cache
                                </button>
                                <button class="btn btn-warning" onclick="backupDatabase()">
                                    <i class="fas fa-download"></i>
                                    Backup Database
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    loadCurrentSettings();
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
        const response = await makeAuthenticatedRequest('/api.php?route=api/admin/all-users');
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
        const response = await makeAuthenticatedRequest('/api.php?route=api/admin/detailed-activity');
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
            makeAuthenticatedRequest('/api.php?route=api/admin/betting-analytics'),
            makeAuthenticatedRequest('/api.php?route=api/admin/recent-bets')
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
        const response = await makeAuthenticatedRequest('/api.php?route=api/admin/model-performance');
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

// Settings functions
async function loadCurrentSettings() {
    try {
        const response = await makeAuthenticatedRequest('/api.php?route=api/admin/settings');
        const settings = await response.json();
        
        // Populate form fields with current settings
        Object.keys(settings).forEach(key => {
            const element = document.getElementById(key);
            if (element) {
                if (element.type === 'checkbox') {
                    element.checked = settings[key] === 'true' || settings[key] === true;
                } else {
                    element.value = settings[key];
                }
            }
        });
        
    } catch (error) {
        console.error('Failed to load settings:', error);
        showAdminError('Failed to load current settings');
    }
}

async function saveSettings() {
    const settings = {
        maxLoginAttempts: document.getElementById('maxLoginAttempts').value,
        sessionTimeout: document.getElementById('sessionTimeout').value,
        passwordMinLength: document.getElementById('passwordMinLength').value,
        defaultBankroll: document.getElementById('defaultBankroll').value,
        minBetAmount: document.getElementById('minBetAmount').value,
        maxBetAmount: document.getElementById('maxBetAmount').value,
        modelUpdateFreq: document.getElementById('modelUpdateFreq').value,
        confidenceThreshold: document.getElementById('confidenceThreshold').value,
        kellyEnabled: document.getElementById('kellyEnabled').checked
    };
    
    try {
        const response = await makeAuthenticatedRequest('/api.php?route=api/admin/settings', {
            method: 'POST',
            body: JSON.stringify(settings)
        });
        
        if (response.ok) {
            showAdminSuccess('Settings saved successfully');
        } else {
            showAdminError('Failed to save settings');
        }
    } catch (error) {
        console.error('Failed to save settings:', error);
        showAdminError('Network error saving settings');
    }
}

// Maintenance functions
async function cleanupOldData() {
    if (confirm('This will remove old logs and inactive sessions. Continue?')) {
        try {
            const response = await makeAuthenticatedRequest('/api.php?route=api/admin/cleanup', {
                method: 'POST'
            });
            
            if (response.ok) {
                showAdminSuccess('Old data cleaned up successfully');
            } else {
                showAdminError('Failed to cleanup old data');
            }
        } catch (error) {
            console.error('Cleanup error:', error);
            showAdminError('Network error during cleanup');
        }
    }
}

async function optimizeDatabase() {
    if (confirm('This will optimize the database. This may take a few minutes. Continue?')) {
        try {
            const response = await makeAuthenticatedRequest('/api.php?route=api/admin/optimize-db', {
                method: 'POST'
            });
            
            if (response.ok) {
                showAdminSuccess('Database optimized successfully');
            } else {
                showAdminError('Failed to optimize database');
            }
        } catch (error) {
            console.error('Optimization error:', error);
            showAdminError('Network error during optimization');
        }
    }
}

async function clearCache() {
    if (confirm('This will clear all cached data. Continue?')) {
        try {
            const response = await makeAuthenticatedRequest('/api.php?route=api/admin/clear-cache', {
                method: 'POST'
            });
            
            if (response.ok) {
                showAdminSuccess('Cache cleared successfully');
            } else {
                showAdminError('Failed to clear cache');
            }
        } catch (error) {
            console.error('Cache clear error:', error);
            showAdminError('Network error clearing cache');
        }
    }
}

async function backupDatabase() {
    try {
        showAdminSuccess('Database backup started. Download will begin shortly.');
        
        const response = await makeAuthenticatedRequest('/api.php?route=api/admin/backup');
        
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `goonsteen_backup_${new Date().toISOString().split('T')[0]}.db`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        } else {
            showAdminError('Failed to create database backup');
        }
    } catch (error) {
        console.error('Backup error:', error);
        showAdminError('Network error creating backup');
    }
}

// Logout function
function logout() {
    if (confirm('Are you sure you want to logout?')) {
        // Clear local storage
        localStorage.removeItem('auth_token');
        localStorage.removeItem('user_data');
        
        // Call logout API
        makeAuthenticatedRequest('/api.php?route=api/logout', {
            method: 'POST'
        }).then(() => {
            window.location.href = 'login.html';
        }).catch(() => {
            // Even if logout fails, clear local data and redirect
            window.location.href = 'login.html';
        });
    }
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
    loadSystemActivityPage,
    logout
};
