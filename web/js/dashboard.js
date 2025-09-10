// Dashboard JavaScript

// Import auth functions
function getAuthToken() {
    return localStorage.getItem('auth_token');
}

function getUserData() {
    const userData = localStorage.getItem('user_data');
    return userData ? JSON.parse(userData) : null;
}

function clearAuthData() {
    localStorage.removeItem('auth_token');
    localStorage.removeItem('user_data');
}

function isAuthenticated() {
    const token = getAuthToken();
    if (!token) return false;
    
    try {
        const payload = JSON.parse(atob(token.split('.')[1]));
        return payload.exp * 1000 > Date.now();
    } catch {
        return false;
    }
}

async function makeAuthenticatedRequest(url, options = {}) {
    const token = getAuthToken();
    console.log('Making authenticated request to:', url);
    console.log('Token exists:', !!token);
    
    if (!token) {
        console.error('No authentication token found');
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
    
    try {
        const response = await fetch(url, mergedOptions);
        console.log('Response status for', url, ':', response.status);
        
        if (response.status === 401) {
            console.error('Unauthorized - clearing auth data');
            clearAuthData();
            window.location.href = '/login.html';
            return null;
        }
        
        if (!response.ok) {
            console.error('Request failed:', response.status, response.statusText);
            const errorText = await response.text();
            console.error('Error details:', errorText);
        }
        
        return response;
    } catch (error) {
        console.error('Network error for', url, ':', error);
        throw error;
    }
}

document.addEventListener('DOMContentLoaded', function() {
    console.log('Dashboard DOM loaded');
    
    // Check if user is authenticated first
    if (!isAuthenticated()) {
        console.log('User not authenticated, redirecting to login');
        window.location.href = 'login.html';
        return;
    }
    
    console.log('User is authenticated, initializing dashboard');
    
    // Show stored user data immediately
    const storedUserData = getUserData();
    console.log('Stored user data:', storedUserData);
    
    if (storedUserData) {
        updateUserInfo(storedUserData);
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
    // Immediately show user data if available
    const storedUserData = getUserData();
    if (storedUserData) {
        console.log('Immediately showing stored user data:', storedUserData);
        updateUserInfo(storedUserData);
    }
    
    // Initialize with default data immediately to prevent loading states
    console.log('Setting initial default data...');
    updateOverviewCards({
        active_bets: 0,
        pending_results: 0,
        week_wins: 0,
        week_losses: 0,
        profit_loss: 0,
        roi: 0
    });
    
    updateActivityList([]);
    updateGamesList([]);
    
    // Then load real data
    loadUserData();
    loadDashboardData();
    initPageNavigation();
    initMobileNavigation();
    
    // Fallback timeout to prevent eternal loading
    setTimeout(() => {
        console.log('Fallback timeout triggered - ensuring data is displayed');
        
        // Check if any elements are still showing "Loading..."
        const loadingElements = document.querySelectorAll('[id*="Value"], [id*="Change"]');
        loadingElements.forEach(element => {
            if (element.textContent.includes('Loading...')) {
                console.log('Found stuck loading element:', element.id);
                
                if (element.id.includes('bankroll')) {
                    element.textContent = '$0.00';
                } else if (element.id.includes('Value')) {
                    element.textContent = '0';
                } else if (element.id.includes('Change')) {
                    element.className = 'card-change neutral';
                    element.innerHTML = '<i class="fas fa-info-circle"></i> No data yet';
                }
            }
        });
        
        // Check activity list
        const activityList = document.getElementById('activityList');
        if (activityList && activityList.textContent.includes('Loading...')) {
            updateActivityList([]);
        }
        
        // Check user info
        const userName = document.getElementById('userName');
        const userStatus = document.getElementById('userStatus');
        
        if (userName && userName.textContent === 'Loading...') {
            const storedUserData = getUserData();
            if (storedUserData) {
                updateUserInfo(storedUserData);
            } else {
                userName.textContent = 'User';
                if (userStatus) userStatus.textContent = 'Free Member';
            }
        }
        
    }, 3000); // 3 second timeout
}

// Check if user is authenticated
async function checkAuthentication() {
    try {
        // First, use stored user data immediately for faster loading
        const storedUserData = getUserData();
        if (storedUserData) {
            console.log('Using stored user data immediately:', storedUserData);
            updateUserInfo(storedUserData);
            
            if (storedUserData.is_admin) {
                showAdminFeatures();
            }
        }
        
        // Then verify with server
        const response = await makeAuthenticatedRequest('/api/session');
        console.log('Session response status:', response?.status);
        
        if (response && response.ok) {
            const data = await response.json();
            console.log('Session data:', data);
            
            if (!data.authenticated) {
                window.location.href = 'login.html';
                return;
            }
            
            // Update user info in sidebar with fresh data
            updateUserInfo(data.user);
            
            // If admin, show admin features
            if (data.user.is_admin) {
                showAdminFeatures();
            }
        } else {
            console.error('Session check failed:', response?.status);
            // If session check fails but we have stored data, continue with stored data
            if (!storedUserData) {
                window.location.href = 'login.html';
            }
        }
        
    } catch (error) {
        console.error('Authentication check failed:', error);
        // Only redirect if we don't have stored user data
        const storedUserData = getUserData();
        if (!storedUserData) {
            window.location.href = 'login.html';
        }
    }
}

// Update user info in sidebar
function updateUserInfo(user) {
    const userName = document.getElementById('userName');
    const userStatus = document.getElementById('userStatus');
    const userAvatar = document.getElementById('userAvatar');
    
    if (userName) {
        userName.textContent = `${user.first_name} ${user.last_name}`;
    }
    
    if (userStatus) {
        const statusText = user.subscription_type === 'premium' ? 'Premium Member' : 
                          user.subscription_type === 'pro' ? 'Pro Member' : 'Free Member';
        userStatus.textContent = statusText;
    }
    
    if (userAvatar) {
        if (user.avatar) {
            userAvatar.innerHTML = `<img src="${user.avatar}" alt="Avatar" style="width: 100%; height: 100%; border-radius: 50%; object-fit: cover;">`;
        } else {
            // Create initials from first and last name
            const initials = `${user.first_name.charAt(0).toUpperCase()}${user.last_name.charAt(0).toUpperCase()}`;
            userAvatar.innerHTML = initials;
        }
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
        console.log('Loading user data...');
        const response = await makeAuthenticatedRequest('/api/user/profile');
        console.log('User profile response status:', response?.status);
        
        if (response && response.ok) {
            const userData = await response.json();
            console.log('User data:', userData);
            updateDashboardWithUserData(userData);
            updateUserInfo(userData); // Update sidebar with real user info
        } else {
            console.error('User profile request failed:', response?.status);
            // Try to get user data from stored session data
            const storedUserData = getUserData();
            if (storedUserData) {
                console.log('Using stored user data:', storedUserData);
                updateUserInfo(storedUserData);
            }
        }
    } catch (error) {
        console.error('Failed to load user data:', error);
        // Fallback to stored user data
        const storedUserData = getUserData();
        if (storedUserData) {
            console.log('Using stored user data as fallback:', storedUserData);
            updateUserInfo(storedUserData);
        }
    }
}

// Update dashboard with user data
function updateDashboardWithUserData(userData) {
    // Update bankroll value
    const bankrollValue = document.getElementById('bankrollValue');
    if (bankrollValue) {
        const balance = userData.total_balance || 0;
        bankrollValue.textContent = formatCurrency(balance);
    }
    
    // Update bankroll change indicator
    const bankrollChange = document.getElementById('bankrollChange');
    if (bankrollChange) {
        if (userData.total_balance > 0) {
            const profitLoss = userData.total_profit_loss || 0;
            const percentage = userData.total_balance > 0 ? (profitLoss / userData.total_balance * 100) : 0;
            
            if (percentage > 0) {
                bankrollChange.className = 'card-change positive';
                bankrollChange.innerHTML = `<i class="fas fa-arrow-up"></i> +${percentage.toFixed(1)}% profit`;
            } else if (percentage < 0) {
                bankrollChange.className = 'card-change negative';
                bankrollChange.innerHTML = `<i class="fas fa-arrow-down"></i> ${percentage.toFixed(1)}% loss`;
            } else {
                bankrollChange.className = 'card-change neutral';
                bankrollChange.innerHTML = `<i class="fas fa-edit"></i> Click to update`;
            }
        } else {
            bankrollChange.className = 'card-change neutral';
            bankrollChange.innerHTML = `<i class="fas fa-edit"></i> Click to update`;
        }
    }
}

// Load dashboard data
async function loadDashboardData() {
    try {
        console.log('Loading dashboard data...');
        
        // Load overview data
        try {
            const overviewResponse = await makeAuthenticatedRequest('/api/dashboard/overview');
            console.log('Overview response status:', overviewResponse.status);
            
            if (overviewResponse && overviewResponse.ok) {
                const overview = await overviewResponse.json();
                console.log('Overview data:', overview);
                updateOverviewCards(overview);
            } else {
                console.error('Overview request failed:', overviewResponse?.status);
            }
        } catch (error) {
            console.error('Overview error:', error);
            // Initialize with defaults for overview
            updateOverviewCards({
                active_bets: 0,
                pending_results: 0,
                week_wins: 0,
                week_losses: 0,
                profit_loss: 0,
                roi: 0
            });
        }
        
        // Load games data
        try {
            const gamesResponse = await makeAuthenticatedRequest('/api/dashboard/games');
            console.log('Games response status:', gamesResponse?.status);
            
            if (gamesResponse && gamesResponse.ok) {
                const games = await gamesResponse.json();
                console.log('Games data:', games);
                updateGamesList(games);
            } else {
                console.error('Games request failed:', gamesResponse?.status);
                updateGamesList([]);
            }
        } catch (error) {
            console.error('Games error:', error);
            updateGamesList([]);
        }
        
        // Load activity data
        try {
            const activityResponse = await makeAuthenticatedRequest('/api/dashboard/activity');
            console.log('Activity response status:', activityResponse?.status);
            
            if (activityResponse && activityResponse.ok) {
                const activity = await activityResponse.json();
                console.log('Activity data:', activity);
                updateActivityList(activity);
            } else {
                console.error('Activity request failed:', activityResponse?.status);
                updateActivityList([]);
            }
        } catch (error) {
            console.error('Activity error:', error);
            updateActivityList([]);
        }
        
    } catch (error) {
        console.error('Failed to load dashboard data:', error);
        showErrorState();
    }
}

// Update overview cards
function updateOverviewCards(data) {
    console.log('Updating overview cards with data:', data);
    
    // Update Active Bets
    const activeBetsValue = document.getElementById('activeBetsValue');
    const activeBetsChange = document.getElementById('activeBetsChange');
    
    if (activeBetsValue) {
        activeBetsValue.textContent = data?.active_bets || 0;
    }
    if (activeBetsChange) {
        activeBetsChange.className = 'card-change neutral';
        activeBetsChange.innerHTML = `<i class="fas fa-clock"></i> ${data?.pending_results || 0} pending results`;
    }
    
    // Update This Week
    const thisWeekValue = document.getElementById('thisWeekValue');
    const thisWeekChange = document.getElementById('thisWeekChange');
    const totalWeekGames = (data?.week_wins || 0) + (data?.week_losses || 0);
    
    if (thisWeekValue) {
        thisWeekValue.textContent = totalWeekGames > 0 ? `${data.week_wins || 0}-${data.week_losses || 0}` : '0-0';
    }
    
    if (thisWeekChange) {
        if (totalWeekGames > 0) {
            const winRate = (data.week_wins / totalWeekGames * 100).toFixed(1);
            updateChangeIndicator(thisWeekChange, winRate, '%', 'win rate');
        } else {
            thisWeekChange.className = 'card-change neutral';
            thisWeekChange.innerHTML = `<i class="fas fa-info-circle"></i> No bets this week`;
        }
    }
    
    // Update Profit/Loss
    const profitLossValue = document.getElementById('profitLossValue');
    const profitLossChange = document.getElementById('profitLossChange');
    
    if (profitLossValue) profitLossValue.textContent = formatCurrency(data?.profit_loss || 0);
    if (profitLossChange) {
        if (data?.roi && data.roi !== 0) {
            updateChangeIndicator(profitLossChange, data.roi, '%', 'ROI');
        } else {
            profitLossChange.className = 'card-change neutral';
            profitLossChange.innerHTML = `<i class="fas fa-chart-line"></i> Track bets to see ROI`;
        }
    }
}

// Initialize with default data if API fails
function initializeWithDefaults() {
    console.log('Initializing with default data...');
    
    // Set default overview data
    updateOverviewCards({
        active_bets: 0,
        pending_results: 0,
        week_wins: 0,
        week_losses: 0,
        profit_loss: 0,
        roi: 0
    });
    
    // Set default activity
    updateActivityList([]);
    
    // Set default games
    updateGamesList([]);
}

// Update games list
function updateGamesList(games) {
    const gamesList = document.getElementById('gamesList');
    if (!gamesList) return;
    
    console.log('Updating games list with:', games);
    gamesList.innerHTML = '';
    
    if (!games || games.length === 0) {
        gamesList.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-basketball-ball"></i>
                <p>No games scheduled for today. Check back tomorrow!</p>
            </div>
        `;
        return;
    }
    
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
    const activityList = document.getElementById('activityList');
    if (!activityList) return;
    
    activityList.innerHTML = '';
    
    if (activities.length === 0) {
        activityList.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-history"></i>
                <p>No recent activity. Start tracking your bets!</p>
            </div>
        `;
        return;
    }
    
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
            
            // Update active state for sidebar navigation
            document.querySelectorAll('.nav-item').forEach(item => item.classList.remove('active'));
            this.closest('.nav-item').classList.add('active');
            
            // Update active state for mobile navigation
            document.querySelectorAll('.mobile-nav-item').forEach(item => {
                item.classList.remove('active');
                if (item.getAttribute('data-page') === page) {
                    item.classList.add('active');
                }
            });
        });
    });
    
    // Set initial active state based on current page
    const currentPage = getCurrentPage();
    setActiveNavigation(currentPage);
}

// Get current page from URL or default to dashboard
function getCurrentPage() {
    const hash = window.location.hash;
    if (hash) {
        return hash.substring(1);
    }
    return 'dashboard';
}

// Set active navigation state
function setActiveNavigation(page) {
    // Update sidebar navigation
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.remove('active');
        const link = item.querySelector('.nav-link[data-page]');
        if (link && link.getAttribute('data-page') === page) {
            item.classList.add('active');
        }
    });
    
    // Update mobile navigation
    document.querySelectorAll('.mobile-nav-item').forEach(item => {
        item.classList.remove('active');
        if (item.getAttribute('data-page') === page) {
            item.classList.add('active');
        }
    });
}

// Navigate to page
function navigateToPage(page) {
    const content = document.getElementById('dashboardContent');
    const pageTitle = document.querySelector('.page-title');
    const pageSubtitle = document.querySelector('.page-subtitle');
    
    // Update URL without hash for dashboard
    if (page === 'dashboard' || !page) {
        history.pushState({page: 'dashboard'}, '', window.location.pathname);
    } else {
        history.pushState({page}, '', `#${page}`);
    }
    
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
        case 'help':
            pageTitle.textContent = 'Help & Support';
            pageSubtitle.textContent = 'Get help and contact support';
            loadHelpPage();
            break;
        default:
            pageTitle.textContent = 'Dashboard';
            pageSubtitle.textContent = 'Welcome back! Here\'s your betting overview.';
            loadDashboardPage();
    }
}

// Load dashboard page content
function loadDashboardPage() {
    const content = document.getElementById('dashboardContent');
    content.innerHTML = `
        <!-- Overview Cards -->
        <div class="overview-cards">
            <div class="overview-card bankroll-card" onclick="showBankrollModal()">
                <div class="card-header">
                    <div class="card-title">Total Bankroll</div>
                    <div class="card-icon">
                        <i class="fas fa-wallet"></i>
                    </div>
                </div>
                <div class="card-value" id="bankrollValue">$0.00</div>
                <div class="card-change positive" id="bankrollChange">
                    <i class="fas fa-edit"></i>
                    Click to update
                </div>
            </div>

            <div class="overview-card">
                <div class="card-header">
                    <div class="card-title">Active Bets</div>
                    <div class="card-icon">
                        <i class="fas fa-chart-line"></i>
                    </div>
                </div>
                <div class="card-value" id="activeBetsValue">Loading...</div>
                <div class="card-change neutral" id="activeBetsChange">
                    <i class="fas fa-clock"></i>
                    Loading...
                </div>
            </div>

            <div class="overview-card">
                <div class="card-header">
                    <div class="card-title">This Week</div>
                    <div class="card-icon">
                        <i class="fas fa-trophy"></i>
                    </div>
                </div>
                <div class="card-value" id="thisWeekValue">Loading...</div>
                <div class="card-change neutral" id="thisWeekChange">
                    <i class="fas fa-clock"></i>
                    Loading...
                </div>
            </div>

            <div class="overview-card">
                <div class="card-header">
                    <div class="card-title">Profit/Loss</div>
                    <div class="card-icon">
                        <i class="fas fa-chart-pie"></i>
                    </div>
                </div>
                <div class="card-value" id="profitLossValue">Loading...</div>
                <div class="card-change neutral" id="profitLossChange">
                    <i class="fas fa-clock"></i>
                    Loading...
                </div>
            </div>
        </div>

        <!-- Main Dashboard Grid -->
        <div class="dashboard-grid">
            <!-- Today's Games -->
            <div class="dashboard-section">
                <div class="section-header">
                    <h2 class="section-title">Today's Games</h2>
                    <div class="section-actions">
                        <button class="btn btn-outline btn-sm">
                            <i class="fas fa-filter"></i>
                            Filter
                        </button>
                        <button class="btn btn-primary btn-sm">
                            <i class="fas fa-refresh"></i>
                            Refresh
                        </button>
                    </div>
                </div>
                
                <div class="games-list" id="gamesList">
                    <div class="loading-message">
                        <i class="fas fa-clock"></i>
                        Loading today's games...
                    </div>
                </div>
            </div>

            <!-- Recent Activity -->
            <div class="dashboard-section">
                <div class="section-header">
                    <h2 class="section-title">Recent Activity</h2>
                    <a href="#history" class="section-link">View All</a>
                </div>
                
                <div class="activity-list" id="activityList">
                    <div class="loading-message">
                        <i class="fas fa-clock"></i>
                        Loading recent activity...
                    </div>
                </div>
            </div>

            <!-- Performance Chart -->
            <div class="dashboard-section chart-section">
                <div class="section-header">
                    <h2 class="section-title">Performance Overview</h2>
                    <div class="chart-controls">
                        <button class="chart-control active" data-period="7d">7D</button>
                        <button class="chart-control" data-period="1m">1M</button>
                        <button class="chart-control" data-period="3m">3M</button>
                        <button class="chart-control" data-period="1y">1Y</button>
                    </div>
                </div>
                
                <div class="chart-container">
                    <canvas id="performanceChart"></canvas>
                </div>
            </div>

            <!-- Quick Stats -->
            <div class="dashboard-section stats-section">
                <div class="section-header">
                    <h2 class="section-title">Quick Stats</h2>
                </div>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-icon">
                            <i class="fas fa-percentage"></i>
                        </div>
                        <div class="stat-content">
                            <div class="stat-value">68.9%</div>
                            <div class="stat-label">Overall Win Rate</div>
                        </div>
                    </div>

                    <div class="stat-card">
                        <div class="stat-icon">
                            <i class="fas fa-chart-line"></i>
                        </div>
                        <div class="stat-content">
                            <div class="stat-value">127</div>
                            <div class="stat-label">Total Bets</div>
                        </div>
                    </div>

                    <div class="stat-card">
                        <div class="stat-icon">
                            <i class="fas fa-fire"></i>
                        </div>
                        <div class="stat-content">
                            <div class="stat-value">7</div>
                            <div class="stat-label">Win Streak</div>
                        </div>
                    </div>

                    <div class="stat-card">
                        <div class="stat-icon">
                            <i class="fas fa-star"></i>
                        </div>
                        <div class="stat-content">
                            <div class="stat-value">4.8</div>
                            <div class="stat-label">Avg Units Won</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Reinitialize dashboard components
    loadDashboardData();
    initCharts();
}

// Mobile navigation
function initMobileNavigation() {
    const mobileNavItems = document.querySelectorAll('.mobile-nav-item[data-page]');
    
    mobileNavItems.forEach(item => {
        item.addEventListener('click', function(e) {
            e.preventDefault();
            
            const page = this.getAttribute('data-page');
            navigateToPage(page);
            
            // Update active state for both mobile and sidebar navigation
            setActiveNavigation(page);
        });
    });
}

// Notifications
function initNotifications() {
    // Load notifications immediately
    loadNotifications();
    
    const notificationPanel = document.getElementById('notificationPanel');
    
    // Close notifications when clicking outside
    document.addEventListener('click', function(e) {
        const notificationsBtn = document.querySelector('.notifications-btn');
        if (notificationPanel && !notificationPanel.contains(e.target) && notificationsBtn && !notificationsBtn.contains(e.target)) {
            notificationPanel.classList.remove('active');
        }
    });
}

function toggleNotifications() {
    const notificationPanel = document.getElementById('notificationPanel');
    if (notificationPanel) {
        notificationPanel.classList.toggle('active');
        if (notificationPanel.classList.contains('active')) {
            loadNotifications();
        }
    }
}

// Load notifications
async function loadNotifications() {
    try {
        const response = await makeAuthenticatedRequest('/api/notifications');
        
        if (response && response.ok) {
            const notifications = await response.json();
            updateNotificationsList(notifications);
            updateNotificationBadge(notifications.filter(n => !n.read).length);
        } else {
            // No notifications or error
            updateNotificationsList([]);
            updateNotificationBadge(0);
        }
        
    } catch (error) {
        console.error('Failed to load notifications:', error);
        // Set empty notifications on error
        updateNotificationsList([]);
        updateNotificationBadge(0);
    }
}

// Update notifications list
function updateNotificationsList(notifications) {
    const notificationList = document.getElementById('notificationList');
    if (!notificationList) return;
    
    notificationList.innerHTML = '';
    
    if (!notifications || notifications.length === 0) {
        notificationList.innerHTML = `
            <div class="empty-notifications">
                <i class="fas fa-bell-slash"></i>
                <p>No notifications yet</p>
            </div>
        `;
        return;
    }
    
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

function formatDate(timestamp) {
    return new Date(timestamp).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    });
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

function loadPredictionsPage() {
    const content = document.getElementById('dashboardContent');
    content.innerHTML = `
        <div class="predictions-page">
            <!-- Floating Prediction Orbs Background -->
            <div class="prediction-orbs">
                <div class="orb orb-1"></div>
                <div class="orb orb-2"></div>
                <div class="orb orb-3"></div>
                <div class="orb orb-4"></div>
                <div class="orb orb-5"></div>
            </div>
            
            <!-- Hero Section -->
            <div class="predictions-hero">
                <div class="hero-content">
                    <div class="hero-badge">
                        <i class="fas fa-brain"></i>
                        <span>AI-Powered</span>
                    </div>
                    <h1 class="hero-title">Today's Predictions</h1>
                    <p class="hero-subtitle">Advanced machine learning models analyzing thousands of data points</p>
                </div>
                
                <div class="accuracy-display">
                    <div class="accuracy-circle">
                        <div class="accuracy-value">68.9%</div>
                        <div class="accuracy-label">Model Accuracy</div>
                    </div>
                </div>
            </div>
            
            <!-- Filter Controls -->
            <div class="prediction-controls">
                <div class="control-group">
                    <div class="filter-chip active" data-confidence="all">
                        <i class="fas fa-star"></i>
                        All Predictions
                    </div>
                    <div class="filter-chip" data-confidence="high">
                        <i class="fas fa-fire"></i>
                        High Confidence
                    </div>
                    <div class="filter-chip" data-confidence="medium">
                        <i class="fas fa-chart-line"></i>
                        Medium Confidence
                    </div>
                    <div class="filter-chip" data-confidence="low">
                        <i class="fas fa-info-circle"></i>
                        Low Confidence
                    </div>
                </div>
                
                <div class="control-actions">
                    <button class="btn btn-primary btn-glow" onclick="refreshPredictions()">
                        <i class="fas fa-sync-alt"></i>
                        Refresh Predictions
                    </button>
                </div>
            </div>
            
            <!-- Predictions Grid -->
            <div class="predictions-container">
                <div class="predictions-grid" id="predictionsGrid">
                    <div class="prediction-loading">
                        <div class="loading-spinner-large"></div>
                        <p>Analyzing games with AI...</p>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    loadPredictionsData();
    initPredictionFilters();
}

function loadParlaysPage() {
    const content = document.getElementById('dashboardContent');
    content.innerHTML = `
        <div class="parlays-page">
            <!-- Animated Background -->
            <div class="parlay-background">
                <div class="floating-shapes">
                    <div class="shape shape-1"></div>
                    <div class="shape shape-2"></div>
                    <div class="shape shape-3"></div>
                    <div class="shape shape-4"></div>
                </div>
            </div>
            
            <!-- Hero Section -->
            <div class="parlay-hero">
                <div class="hero-content">
                    <div class="hero-badge pulse">
                        <i class="fas fa-layer-group"></i>
                        <span>Parlay Builder</span>
                    </div>
                    <h1 class="hero-title gradient-text">Build Your Perfect Parlay</h1>
                    <p class="hero-subtitle">Combine multiple bets for bigger payouts with AI-powered correlation analysis</p>
                </div>
                
                <div class="parlay-stats">
                    <div class="stat-bubble">
                        <div class="stat-number">2.5x</div>
                        <div class="stat-label">Avg Payout</div>
                    </div>
                    <div class="stat-bubble">
                        <div class="stat-number">73%</div>
                        <div class="stat-label">Success Rate</div>
                    </div>
                    <div class="stat-bubble">
                        <div class="stat-number">5+</div>
                        <div class="stat-label">Max Legs</div>
                    </div>
                </div>
            </div>
            
            <!-- Parlay Builder Interface -->
            <div class="parlay-builder-container">
                <div class="builder-section games-section">
                    <div class="section-header-fancy">
                        <div class="header-icon">
                            <i class="fas fa-basketball-ball"></i>
                        </div>
                        <div class="header-content">
                            <h3>Available Games</h3>
                            <p>Select games to add to your parlay</p>
                        </div>
                    </div>
                    
                    <div class="games-carousel" id="availableGamesCarousel">
                        <div class="carousel-loading">
                            <div class="loading-pulse"></div>
                            <p>Loading available games...</p>
                        </div>
                    </div>
                </div>
                
                <div class="builder-section parlay-section">
                    <div class="section-header-fancy">
                        <div class="header-icon active">
                            <i class="fas fa-layer-group"></i>
                        </div>
                        <div class="header-content">
                            <h3>Your Parlay</h3>
                            <p id="parlayCount">0 selections</p>
                        </div>
                        <button class="clear-btn" onclick="clearParlay()" style="display: none;">
                            <i class="fas fa-trash"></i>
                        </button>
                    </div>
                    
                    <div class="parlay-slip-container">
                        <div class="parlay-legs" id="parlayLegs">
                            <div class="empty-parlay-state">
                                <div class="empty-icon">
                                    <i class="fas fa-plus-circle"></i>
                                </div>
                                <h4>Start Building</h4>
                                <p>Select games from the left to build your parlay</p>
                            </div>
                        </div>
                        
                        <div class="parlay-calculator" id="parlayCalculator" style="display: none;">
                            <div class="calculator-header">
                                <i class="fas fa-calculator"></i>
                                <span>Parlay Calculator</span>
                            </div>
                            
                            <div class="calculation-rows">
                                <div class="calc-row">
                                    <span>Total Odds:</span>
                                    <span class="odds-value" id="totalOdds">+0</span>
                                </div>
                                <div class="calc-row">
                                    <span>Bet Amount:</span>
                                    <input type="number" id="parlayBetAmount" class="bet-input" placeholder="$0.00" min="1" step="0.01">
                                </div>
                                <div class="calc-row highlight">
                                    <span>Potential Payout:</span>
                                    <span class="payout-value" id="potentialPayout">$0.00</span>
                                </div>
                            </div>
                            
                            <button class="btn btn-primary btn-glow full-width" onclick="trackParlay()">
                                <i class="fas fa-rocket"></i>
                                Track This Parlay
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    loadParlayData();
    initParlayBuilder();
}

function loadAnalyticsPage() {
    const content = document.getElementById('dashboardContent');
    content.innerHTML = `
        <div class="analytics-page">
            <!-- Particle System Background -->
            <div class="analytics-background">
                <div class="particle-system">
                    <div class="particle"></div>
                    <div class="particle"></div>
                    <div class="particle"></div>
                    <div class="particle"></div>
                    <div class="particle"></div>
                    <div class="particle"></div>
                </div>
            </div>
            
            <!-- Analytics Hero -->
            <div class="analytics-hero">
                <div class="hero-content">
                    <div class="hero-badge glow">
                        <i class="fas fa-chart-bar"></i>
                        <span>Analytics Hub</span>
                    </div>
                    <h1 class="hero-title">Performance Analytics</h1>
                    <p class="hero-subtitle">Deep insights into your betting performance with advanced metrics</p>
                </div>
                
                <div class="time-dimension-selector">
                    <div class="time-btn active" data-range="7d">
                        <span class="time-label">7D</span>
                        <span class="time-desc">Last Week</span>
                    </div>
                    <div class="time-btn" data-range="30d">
                        <span class="time-label">30D</span>
                        <span class="time-desc">Last Month</span>
                    </div>
                    <div class="time-btn" data-range="90d">
                        <span class="time-label">90D</span>
                        <span class="time-desc">Quarter</span>
                    </div>
                    <div class="time-btn" data-range="1y">
                        <span class="time-label">1Y</span>
                        <span class="time-desc">Year</span>
                    </div>
                </div>
            </div>
            
            <!-- Performance Dashboard -->
            <div class="analytics-dashboard">
                <!-- Key Metrics Row -->
                <div class="metrics-showcase">
                    <div class="metric-card primary">
                        <div class="metric-icon">
                            <i class="fas fa-percentage"></i>
                        </div>
                        <div class="metric-content">
                            <div class="metric-value" id="analyticsWinRate">0%</div>
                            <div class="metric-label">Win Rate</div>
                            <div class="metric-trend positive">
                                <i class="fas fa-arrow-up"></i>
                                <span>+2.3%</span>
                            </div>
                        </div>
                        <div class="metric-sparkline">
                            <canvas id="winRateSparkline" width="100" height="30"></canvas>
                        </div>
                    </div>
                    
                    <div class="metric-card success">
                        <div class="metric-icon">
                            <i class="fas fa-dollar-sign"></i>
                        </div>
                        <div class="metric-content">
                            <div class="metric-value" id="analyticsTotalProfit">$0</div>
                            <div class="metric-label">Total Profit</div>
                            <div class="metric-trend positive">
                                <i class="fas fa-arrow-up"></i>
                                <span>+15.2%</span>
                            </div>
                        </div>
                        <div class="metric-sparkline">
                            <canvas id="profitSparkline" width="100" height="30"></canvas>
                        </div>
                    </div>
                    
                    <div class="metric-card info">
                        <div class="metric-icon">
                            <i class="fas fa-chart-line"></i>
                        </div>
                        <div class="metric-content">
                            <div class="metric-value" id="analyticsTotalBets">0</div>
                            <div class="metric-label">Total Bets</div>
                            <div class="metric-trend neutral">
                                <i class="fas fa-minus"></i>
                                <span>No change</span>
                            </div>
                        </div>
                        <div class="metric-sparkline">
                            <canvas id="betsSparkline" width="100" height="30"></canvas>
                        </div>
                    </div>
                    
                    <div class="metric-card warning">
                        <div class="metric-icon">
                            <i class="fas fa-fire"></i>
                        </div>
                        <div class="metric-content">
                            <div class="metric-value" id="analyticsBestStreak">0</div>
                            <div class="metric-label">Best Streak</div>
                            <div class="metric-trend positive">
                                <i class="fas fa-arrow-up"></i>
                                <span>Personal Best</span>
                            </div>
                        </div>
                        <div class="metric-sparkline">
                            <canvas id="streakSparkline" width="100" height="30"></canvas>
                        </div>
                    </div>
                </div>
                
                <!-- Main Chart Section -->
                <div class="chart-showcase">
                    <div class="chart-container-fancy">
                        <div class="chart-header">
                            <h3>Performance Over Time</h3>
                            <div class="chart-controls">
                                <button class="chart-btn active" data-chart="profit">Profit/Loss</button>
                                <button class="chart-btn" data-chart="winrate">Win Rate</button>
                                <button class="chart-btn" data-chart="volume">Bet Volume</button>
                            </div>
                        </div>
                        <div class="chart-canvas-container">
                            <canvas id="mainAnalyticsChart"></canvas>
                        </div>
                    </div>
                </div>
                
                <!-- Insights Section -->
                <div class="insights-section">
                    <div class="insights-header">
                        <i class="fas fa-lightbulb"></i>
                        <h3>AI Insights</h3>
                    </div>
                    
                    <div class="insights-grid">
                        <div class="insight-card">
                            <div class="insight-icon">🎯</div>
                            <div class="insight-content">
                                <h4>Betting Pattern</h4>
                                <p>You perform best on games with 70%+ confidence. Consider focusing on high-confidence predictions.</p>
                            </div>
                        </div>
                        
                        <div class="insight-card">
                            <div class="insight-icon">📈</div>
                            <div class="insight-content">
                                <h4>Optimal Bet Size</h4>
                                <p>Your Kelly Criterion suggests betting 2-4% of bankroll for optimal long-term growth.</p>
                            </div>
                        </div>
                        
                        <div class="insight-card">
                            <div class="insight-icon">⚡</div>
                            <div class="insight-content">
                                <h4>Best Time to Bet</h4>
                                <p>Your win rate is highest on evening games. Consider this timing for future bets.</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    loadAnalyticsData();
    initAnalyticsInteractions();
}

function loadHistoryPage() {
    const content = document.getElementById('dashboardContent');
    content.innerHTML = `
        <div class="history-page">
            <div class="page-header">
                <h2>Betting History</h2>
                <div class="history-filters">
                    <select id="statusFilter" class="form-select">
                        <option value="">All Status</option>
                        <option value="won">Won</option>
                        <option value="lost">Lost</option>
                        <option value="pending">Pending</option>
                    </select>
                    <select id="periodFilter" class="form-select">
                        <option value="all">All Time</option>
                        <option value="7d">Last 7 Days</option>
                        <option value="30d">Last 30 Days</option>
                        <option value="90d">Last 90 Days</option>
                    </select>
                </div>
            </div>
            
            <div class="history-content">
                <div class="history-table">
                    <table class="bets-table">
                        <thead>
                            <tr>
                                <th>Date</th>
                                <th>Game</th>
                                <th>Bet Type</th>
                                <th>Amount</th>
                                <th>Odds</th>
                                <th>Status</th>
                                <th>Payout</th>
                            </tr>
                        </thead>
                        <tbody id="historyTableBody">
                            <tr><td colspan="7" class="loading-message">Loading betting history...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    `;
    
    loadBettingHistory();
}

function loadBankrollPage() {
    const content = document.getElementById('dashboardContent');
    content.innerHTML = `
        <div class="bankroll-page">
            <div class="page-header">
                <h2>Bankroll Management</h2>
                <button class="btn btn-primary btn-sm" onclick="showBankrollModal()">
                    <i class="fas fa-edit"></i>
                    Update Balance
                </button>
            </div>
            
            <div class="bankroll-overview">
                <div class="bankroll-card-large">
                    <div class="bankroll-header">
                        <h3>Current Bankroll</h3>
                        <div class="bankroll-amount" id="largeBankrollAmount">$0.00</div>
                    </div>
                    <div class="bankroll-breakdown">
                        <div class="breakdown-item">
                            <span>Available:</span>
                            <span id="availableBalance">$0.00</span>
                        </div>
                        <div class="breakdown-item">
                            <span>In Play:</span>
                            <span id="reservedBalance">$0.00</span>
                        </div>
                        <div class="breakdown-item">
                            <span>Daily Limit:</span>
                            <span id="dailyLimit">$0.00</span>
                        </div>
                    </div>
                </div>
                
                <div class="bankroll-chart">
                    <h3>Bankroll History</h3>
                    <canvas id="bankrollChart"></canvas>
                </div>
            </div>
        </div>
    `;
    
    loadBankrollData();
}

function loadSettingsPage() {
    const content = document.getElementById('dashboardContent');
    content.innerHTML = `
        <div class="settings-page">
            <div class="page-header">
                <h2>Account Settings</h2>
                <button class="btn btn-primary btn-sm" onclick="saveUserSettings()">
                    <i class="fas fa-save"></i>
                    Save Changes
                </button>
            </div>
            
            <div class="settings-sections">
                <div class="setting-section">
                    <h3>Profile Information</h3>
                    <form id="profileForm">
                        <div class="form-row">
                            <div class="form-group">
                                <label for="firstName">First Name</label>
                                <input type="text" id="firstName" class="form-input" required>
                            </div>
                            <div class="form-group">
                                <label for="lastName">Last Name</label>
                                <input type="text" id="lastName" class="form-input" required>
                            </div>
                        </div>
                        <div class="form-group">
                            <label for="email">Email Address</label>
                            <input type="email" id="email" class="form-input" required>
                        </div>
                    </form>
                </div>
                
                <div class="setting-section">
                    <h3>Betting Preferences</h3>
                    <form id="bettingForm">
                        <div class="form-group">
                            <label for="defaultBetAmount">Default Bet Amount</label>
                            <input type="number" id="defaultBetAmount" class="form-input" step="0.01" min="1">
                        </div>
                        <div class="form-group">
                            <div class="checkbox-group">
                                <input type="checkbox" id="kellyEnabled">
                                <label for="kellyEnabled">Enable Kelly Criterion Suggestions</label>
                            </div>
                        </div>
                        <div class="form-group">
                            <div class="checkbox-group">
                                <input type="checkbox" id="emailNotifications">
                                <label for="emailNotifications">Email Notifications</label>
                            </div>
                        </div>
                    </form>
                </div>
            </div>
        </div>
    `;
    
    loadUserSettings();
}

function loadHelpPage() {
    const content = document.getElementById('dashboardContent');
    content.innerHTML = `
        <div class="help-page">
            <div class="help-sections">
                <div class="help-section">
                    <div class="help-header">
                        <i class="fas fa-question-circle"></i>
                        <h3>Frequently Asked Questions</h3>
                    </div>
                    
                    <div class="faq-list">
                        <div class="faq-item">
                            <div class="faq-question" onclick="toggleFaq(this)">
                                <span>How does the AI prediction system work?</span>
                                <i class="fas fa-chevron-down"></i>
                            </div>
                            <div class="faq-answer">
                                <p>Our AI system uses ensemble machine learning models including XGBoost, Neural Networks, and advanced statistical analysis to predict NBA game outcomes. The models analyze thousands of data points including player statistics, team performance, historical trends, and more.</p>
                            </div>
                        </div>
                        
                        <div class="faq-item">
                            <div class="faq-question" onclick="toggleFaq(this)">
                                <span>What is Kelly Criterion and how does it help?</span>
                                <i class="fas fa-chevron-down"></i>
                            </div>
                            <div class="faq-answer">
                                <p>Kelly Criterion is a mathematical formula used to determine optimal bet sizing based on your bankroll and the probability of winning. It helps maximize long-term growth while minimizing risk of ruin.</p>
                            </div>
                        </div>
                        
                        <div class="faq-item">
                            <div class="faq-question" onclick="toggleFaq(this)">
                                <span>How do I track my external sportsbook balance?</span>
                                <i class="fas fa-chevron-down"></i>
                            </div>
                            <div class="faq-answer">
                                <p>Click on the "Total Bankroll" card on your dashboard to update your current sportsbook account balance. This helps our system provide accurate Kelly Criterion suggestions and track your performance.</p>
                            </div>
                        </div>
                        
                        <div class="faq-item">
                            <div class="faq-question" onclick="toggleFaq(this)">
                                <span>Are my funds safe? Do you handle real money?</span>
                                <i class="fas fa-chevron-down"></i>
                            </div>
                            <div class="faq-answer">
                                <p>GoonSteen is an analytical platform only. We do not handle real money transactions. You place bets with your own sportsbook accounts and use our platform to track performance and get AI predictions.</p>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="help-section">
                    <div class="help-header">
                        <i class="fas fa-headset"></i>
                        <h3>Contact Support</h3>
                    </div>
                    
                    <div class="contact-options">
                        <div class="contact-option">
                            <div class="contact-icon">
                                <i class="fas fa-envelope"></i>
                            </div>
                            <div class="contact-info">
                                <h4>Email Support</h4>
                                <p>support@goonsteen.com</p>
                                <span>Response within 24 hours</span>
                            </div>
                        </div>
                        
                        <div class="contact-option">
                            <div class="contact-icon">
                                <i class="fas fa-comments"></i>
                            </div>
                            <div class="contact-info">
                                <h4>Live Chat</h4>
                                <p>Available 24/7</p>
                                <button class="btn btn-primary btn-sm" onclick="openLiveChat()">Start Chat</button>
                            </div>
                        </div>
                        
                        <div class="contact-option">
                            <div class="contact-icon">
                                <i class="fas fa-book"></i>
                            </div>
                            <div class="contact-info">
                                <h4>Documentation</h4>
                                <p>Comprehensive guides and tutorials</p>
                                <button class="btn btn-outline btn-sm" onclick="openDocs()">View Docs</button>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="help-section">
                    <div class="help-header">
                        <i class="fas fa-chart-line"></i>
                        <h3>Platform Statistics</h3>
                    </div>
                    
                    <div class="platform-stats">
                        <div class="platform-stat">
                            <div class="stat-value">68.9%</div>
                            <div class="stat-label">Overall Model Accuracy</div>
                        </div>
                        <div class="platform-stat">
                            <div class="stat-value">10K+</div>
                            <div class="stat-label">Predictions Made</div>
                        </div>
                        <div class="platform-stat">
                            <div class="stat-value">1,200+</div>
                            <div class="stat-label">Active Users</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
}

// Action handlers
function viewGameAnalysis(gameId) {
    console.log('Viewing analysis for game:', gameId);
    // Show detailed game analysis
    showGameAnalysisModal(gameId);
}

function placeBet(gameId) {
    // Open bet tracking modal
    showBetTrackingModal(gameId);
}

// Bankroll management
function showBankrollModal() {
    const modal = document.getElementById('bankrollModal');
    if (modal) {
        // Load current bankroll data
        loadCurrentBankroll();
        modal.classList.add('active');
        document.body.style.overflow = 'hidden';
    }
}

async function loadCurrentBankroll() {
    try {
        const response = await makeAuthenticatedRequest('/api/user/bankroll');
        const bankroll = await response.json();
        
        document.getElementById('currentBalance').value = bankroll.total_balance || 0;
        document.getElementById('betLimit').value = bankroll.daily_limit || 100;
        
    } catch (error) {
        console.error('Failed to load bankroll:', error);
    }
}

async function updateBankroll() {
    const currentBalance = document.getElementById('currentBalance').value;
    const betLimit = document.getElementById('betLimit').value;
    
    if (!currentBalance || currentBalance <= 0) {
        alert('Please enter a valid balance');
        return;
    }
    
    try {
        const response = await makeAuthenticatedRequest('/api/user/bankroll', {
            method: 'POST',
            body: JSON.stringify({
                total_balance: parseFloat(currentBalance),
                daily_limit: parseFloat(betLimit) || 100
            })
        });
        
        if (response.ok) {
            closeModal('bankrollModal');
            showSuccessNotification('Bankroll updated successfully!');
            
            // Refresh dashboard data
            loadDashboardData();
        } else {
            alert('Failed to update bankroll');
        }
        
    } catch (error) {
        console.error('Failed to update bankroll:', error);
        alert('Network error updating bankroll');
    }
}

// Bet tracking
function showBetTrackingModal(gameId) {
    const modal = document.getElementById('betTrackingModal');
    if (modal) {
        document.getElementById('gameId').value = gameId;
        modal.classList.add('active');
        document.body.style.overflow = 'hidden';
        
        // Add event listeners for Kelly Criterion calculation
        const betAmountInput = document.getElementById('betAmount');
        const oddsInput = document.getElementById('odds');
        
        function calculateKelly() {
            const amount = parseFloat(betAmountInput.value);
            const odds = oddsInput.value;
            
            if (amount && odds) {
                calculateKellyCriterion(amount, odds, gameId);
            }
        }
        
        betAmountInput.addEventListener('input', calculateKelly);
        oddsInput.addEventListener('input', calculateKelly);
    }
}

async function calculateKellyCriterion(betAmount, odds, gameId) {
    try {
        const response = await makeAuthenticatedRequest('/api/calculate-kelly', {
            method: 'POST',
            body: JSON.stringify({
                game_id: gameId,
                bet_amount: betAmount,
                odds: odds
            })
        });
        
        if (response.ok) {
            const result = await response.json();
            const kellySuggestion = document.getElementById('kellySuggestion');
            const kellyAmount = document.getElementById('kellyAmount');
            
            if (result.kelly_amount > 0) {
                kellyAmount.textContent = result.kelly_amount.toFixed(2);
                kellySuggestion.style.display = 'block';
            } else {
                kellySuggestion.style.display = 'none';
            }
        }
        
    } catch (error) {
        console.error('Kelly calculation error:', error);
    }
}

async function trackBet() {
    const gameId = document.getElementById('gameId').value;
    const betAmount = document.getElementById('betAmount').value;
    const betType = document.getElementById('betType').value;
    const odds = document.getElementById('odds').value;
    
    if (!betAmount || !betType || !odds) {
        alert('Please fill in all fields');
        return;
    }
    
    try {
        const response = await makeAuthenticatedRequest('/api/user/track-bet', {
            method: 'POST',
            body: JSON.stringify({
                game_id: gameId,
                bet_amount: parseFloat(betAmount),
                bet_type: betType,
                odds: odds
            })
        });
        
        if (response.ok) {
            closeModal('betTrackingModal');
            showSuccessNotification('Bet tracked successfully!');
            
            // Refresh dashboard data
            loadDashboardData();
        } else {
            const error = await response.json();
            alert(error.message || 'Failed to track bet');
        }
        
    } catch (error) {
        console.error('Failed to track bet:', error);
        alert('Network error tracking bet');
    }
}

// Show success notification
function showSuccessNotification(message) {
    const notification = document.createElement('div');
    notification.className = 'success-notification';
    notification.innerHTML = `
        <div class="notification-content">
            <i class="fas fa-check-circle"></i>
            <span>${message}</span>
        </div>
    `;
    
    // Add styles if not already added
    if (!document.querySelector('#success-notification-styles')) {
        const style = document.createElement('style');
        style.id = 'success-notification-styles';
        style.textContent = `
            .success-notification {
                position: fixed;
                top: 2rem;
                right: 2rem;
                background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
                border-radius: 12px;
                padding: 1rem 1.5rem;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
                border-left: 4px solid #10b981;
                z-index: 10000;
                animation: slideInRight 0.3s ease;
                color: white;
                backdrop-filter: blur(20px);
            }
            
            .success-notification .notification-content {
                display: flex;
                align-items: center;
                gap: 0.75rem;
            }
            
            .success-notification i {
                color: #10b981;
                font-size: 1.25rem;
            }
            
            @keyframes slideInRight {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
        `;
        document.head.appendChild(style);
    }
    
    document.body.appendChild(notification);
    
    // Auto remove after 3 seconds
    setTimeout(() => {
        if (notification.parentElement) {
            notification.remove();
        }
    }, 3000);
}

// Modal functions
function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.classList.remove('active');
        document.body.style.overflow = '';
        
        // Reset form if it exists
        const form = modal.querySelector('form');
        if (form) {
            form.reset();
        }
        
        // Hide Kelly suggestion
        const kellySuggestion = modal.querySelector('#kellySuggestion');
        if (kellySuggestion) {
            kellySuggestion.style.display = 'none';
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
    if (confirm('Are you sure you want to logout?')) {
        // Clear authentication data immediately
        localStorage.removeItem('auth_token');
        localStorage.removeItem('user_data');
        
        // Call logout API but don't wait for response
        fetch('/api/logout', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${localStorage.getItem('auth_token') || ''}`
            }
        }).catch(() => {
            // Ignore errors - we're logging out anyway
        });
        
        // Immediate redirect
        window.location.href = 'login.html';
    }
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
    closeNotifications,
    showBankrollModal,
    updateBankroll,
    trackBet,
    closeModal
};

// Data loading functions for pages
async function loadPredictionsData() {
    try {
        const response = await makeAuthenticatedRequest('/api/dashboard/games');
        const games = await response.json();
        
        const predictionsList = document.getElementById('predictionsList');
        if (predictionsList) {
            predictionsList.innerHTML = '';
            
            games.forEach(game => {
                const predictionCard = createPredictionCard(game);
                predictionsList.appendChild(predictionCard);
            });
        }
    } catch (error) {
        console.error('Failed to load predictions:', error);
    }
}

async function loadParlayData() {
    // Implementation for parlay data loading
    console.log('Loading parlay data...');
}

async function loadAnalyticsData() {
    try {
        const response = await makeAuthenticatedRequest('/api/user/analytics');
        const analytics = await response.json();
        
        // Update analytics metrics
        document.getElementById('analyticsWinRate').textContent = `${analytics.win_rate || 0}%`;
        document.getElementById('analyticsTotalBets').textContent = analytics.total_bets || 0;
        document.getElementById('analyticsAvgBet').textContent = formatCurrency(analytics.avg_bet || 0);
        document.getElementById('analyticsBestStreak').textContent = analytics.best_streak || 0;
        
    } catch (error) {
        console.error('Failed to load analytics:', error);
    }
}

async function loadBettingHistory() {
    try {
        const response = await makeAuthenticatedRequest('/api/user/betting-history');
        const bets = await response.json();
        
        const tbody = document.getElementById('historyTableBody');
        if (tbody) {
            tbody.innerHTML = '';
            
            if (bets.length === 0) {
                tbody.innerHTML = '<tr><td colspan="7" class="no-data">No betting history found. Start tracking your bets!</td></tr>';
                return;
            }
            
            bets.forEach(bet => {
                const row = createHistoryRow(bet);
                tbody.appendChild(row);
            });
        }
    } catch (error) {
        console.error('Failed to load betting history:', error);
    }
}

async function loadBankrollData() {
    try {
        const response = await makeAuthenticatedRequest('/api/user/bankroll');
        const bankroll = await response.json();
        
        document.getElementById('largeBankrollAmount').textContent = formatCurrency(bankroll.total_balance || 0);
        document.getElementById('availableBalance').textContent = formatCurrency(bankroll.available_balance || 0);
        document.getElementById('reservedBalance').textContent = formatCurrency(bankroll.reserved_balance || 0);
        document.getElementById('dailyLimit').textContent = formatCurrency(bankroll.daily_limit || 100);
        
    } catch (error) {
        console.error('Failed to load bankroll data:', error);
    }
}

async function loadUserSettings() {
    try {
        const response = await makeAuthenticatedRequest('/api/user/profile');
        const user = await response.json();
        
        document.getElementById('firstName').value = user.first_name || '';
        document.getElementById('lastName').value = user.last_name || '';
        document.getElementById('email').value = user.email || '';
        document.getElementById('defaultBetAmount').value = user.default_bet_amount || 50;
        document.getElementById('kellyEnabled').checked = user.kelly_enabled !== false;
        document.getElementById('emailNotifications').checked = user.email_notifications !== false;
        
    } catch (error) {
        console.error('Failed to load user settings:', error);
    }
}

// Helper functions
function createPredictionCard(game) {
    const card = document.createElement('div');
    card.className = 'prediction-card';
    card.innerHTML = `
        <div class="prediction-header">
            <div class="game-teams">${game.away_team.name} @ ${game.home_team.name}</div>
            <div class="confidence-badge ${getConfidenceClass(game.confidence)}">${game.confidence}% Confidence</div>
        </div>
        <div class="prediction-content">
            <div class="prediction-pick">
                <strong>${game.prediction.winner} Win</strong>
                <span>${game.prediction.score}</span>
            </div>
            <button class="btn btn-primary btn-sm" onclick="placeBet('${game.id}')">Track Bet</button>
        </div>
    `;
    return card;
}

function createHistoryRow(bet) {
    const row = document.createElement('tr');
    const statusClass = bet.status === 'won' ? 'positive' : bet.status === 'lost' ? 'negative' : 'neutral';
    
    row.innerHTML = `
        <td>${formatDate(bet.placed_at)}</td>
        <td>${bet.game_info || 'Game'}</td>
        <td>${bet.bet_type}</td>
        <td>${formatCurrency(bet.stake)}</td>
        <td>${bet.odds_display || bet.odds}</td>
        <td><span class="status-badge ${statusClass}">${bet.status}</span></td>
        <td>${formatCurrency(bet.actual_payout || 0)}</td>
    `;
    return row;
}

function refreshPredictions() {
    loadPredictionsData();
}

function clearParlay() {
    // Implementation for clearing parlay
    console.log('Clearing parlay...');
}

function trackParlay() {
    // Implementation for tracking parlay
    console.log('Tracking parlay...');
}

function saveUserSettings() {
    // Implementation for saving user settings
    console.log('Saving user settings...');
}

// Help page functions
function toggleFaq(element) {
    const faqItem = element.closest('.faq-item');
    const answer = faqItem.querySelector('.faq-answer');
    const icon = element.querySelector('i');
    
    faqItem.classList.toggle('active');
    
    if (faqItem.classList.contains('active')) {
        answer.style.maxHeight = answer.scrollHeight + 'px';
        icon.style.transform = 'rotate(180deg)';
    } else {
        answer.style.maxHeight = '0';
        icon.style.transform = 'rotate(0deg)';
    }
}

function openLiveChat() {
    alert('Live chat feature coming soon! Please email support@goonsteen.com for immediate assistance.');
}

function openDocs() {
    window.open('WEB_README.md', '_blank');
}

// Make functions globally available for onclick handlers
window.viewGameAnalysis = viewGameAnalysis;
window.placeBet = placeBet;
window.logout = logout;
window.showBankrollModal = showBankrollModal;
window.updateBankroll = updateBankroll;
window.trackBet = trackBet;
window.closeModal = closeModal;
window.refreshPredictions = refreshPredictions;
window.clearParlay = clearParlay;
window.trackParlay = trackParlay;
window.saveUserSettings = saveUserSettings;
window.toggleFaq = toggleFaq;
window.openLiveChat = openLiveChat;
window.openDocs = openDocs;
window.toggleNotifications = toggleNotifications;

// Additional functions for new pages
function initPredictionFilters() {
    const filterChips = document.querySelectorAll('.filter-chip');
    filterChips.forEach(chip => {
        chip.addEventListener('click', function() {
            filterChips.forEach(c => c.classList.remove('active'));
            this.classList.add('active');
            
            const confidence = this.getAttribute('data-confidence');
            filterPredictions(confidence);
        });
    });
}

function filterPredictions(confidence) {
    console.log('Filtering predictions by confidence:', confidence);
    // Implementation for filtering predictions
}

function initParlayBuilder() {
    const parlayBetInput = document.getElementById('parlayBetAmount');
    if (parlayBetInput) {
        parlayBetInput.addEventListener('input', calculateParlayPayout);
    }
}

function calculateParlayPayout() {
    const betAmount = parseFloat(document.getElementById('parlayBetAmount').value) || 0;
    const totalOdds = document.getElementById('totalOdds').textContent;
    
    // Simple calculation for demo
    let multiplier = 1;
    if (totalOdds.startsWith('+')) {
        multiplier = (parseFloat(totalOdds.substring(1)) / 100) + 1;
    }
    
    const payout = betAmount * multiplier;
    document.getElementById('potentialPayout').textContent = formatCurrency(payout);
}

function initAnalyticsInteractions() {
    const timeBtns = document.querySelectorAll('.time-btn');
    timeBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            timeBtns.forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            
            const range = this.getAttribute('data-range');
            loadAnalyticsForRange(range);
        });
    });
    
    const chartBtns = document.querySelectorAll('.chart-btn');
    chartBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            chartBtns.forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            
            const chartType = this.getAttribute('data-chart');
            loadChartData(chartType);
        });
    });
}

function loadAnalyticsForRange(range) {
    console.log('Loading analytics for range:', range);
    // Implementation for loading analytics data for specific range
}

function loadChartData(chartType) {
    console.log('Loading chart data for type:', chartType);
    // Implementation for loading different chart types
}
