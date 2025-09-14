// Authentication JavaScript - Updated for API.php backend

// Auth token management
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
    
    // Check if token is expired (basic check)
    try {
        const payload = JSON.parse(atob(token.split('.')[1]));
        return payload.exp * 1000 > Date.now();
    } catch {
        return false;
    }
}

// Make authenticated API requests
async function makeAuthenticatedRequest(url, options = {}) {
    const token = getAuthToken();
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
    
    const response = await fetch(url, mergedOptions);
    
    // If unauthorized, redirect to login
    if (response.status === 401) {
        clearAuthData();
        window.location.href = '/login.html';
        return;
    }
    
    return response;
}

document.addEventListener('DOMContentLoaded', function() {
    initAuthForms();
    initPasswordValidation();
    initUsernameValidation();
    initAdminToggle();
});

// Initialize authentication forms
function initAuthForms() {
    const loginForm = document.getElementById('loginForm');
    const signupForm = document.getElementById('signupForm');
    
    if (loginForm) {
        loginForm.addEventListener('submit', handleLogin);
    }
    
    if (signupForm) {
        signupForm.addEventListener('submit', handleSignup);
    }
}

// Handle login form submission
async function handleLogin(e) {
    e.preventDefault();
    
    const form = e.target;
    const formData = new FormData(form);
    const loginData = {
        username: formData.get('username'),
        password: formData.get('password'),
        remember_me: formData.get('remember_me') === 'on'
    };
    
    // Show loading state
    showLoading(form);
    
    try {
        const response = await fetch('/api.php?route=api/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(loginData)
        });
        
        const result = await response.json();
        
        if (response.ok) {
            // Store JWT token and user data
            localStorage.setItem('auth_token', result.token);
            localStorage.setItem('user_data', JSON.stringify(result.user));
        
            
            // Success - redirect to dashboard
            if (result.user.is_admin) {
                window.location.href = 'admin-dashboard.html';
            } else {
                window.location.href = 'dashboard.html';
            }
        } else {
            // Show error
            showError(result.message || 'Invalid username or password');
        }
    } catch (error) {
        console.error('Login error:', error);
        showError('Network error. Please try again.');
    } finally {
        hideLoading(form);
    }
}

// Handle signup form submission
async function handleSignup(e) {
    e.preventDefault();
    
    const form = e.target;
    const formData = new FormData(form);
    
    // Validate form
    if (!validateSignupForm(form)) {
        return;
    }
    
    const signupData = {
        first_name: formData.get('first_name'),
        last_name: formData.get('last_name'),
        username: formData.get('username'),
        email: formData.get('email'),
        password: formData.get('password'),
        age_verification: formData.get('age_verification'),
        terms_accepted: formData.get('terms_accepted') === 'on',
        marketing_emails: formData.get('marketing_emails') === 'on',
        responsible_gambling: formData.get('responsible_gambling') === 'on'
    };
    
    // Show loading state
    showLoading(form);
    
    try {
        const response = await fetch('/api.php?route=api/signup', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(signupData)
        });
        
        const result = await response.json();
        
        if (response.ok) {
            // Show success modal
            showModal('successModal');
        } else {
            // Show error
            showError(result.message || 'Error creating account');
        }
    } catch (error) {
        console.error('Signup error:', error);
        showError('Network error. Please try again.');
    } finally {
        hideLoading(form);
    }
}

// Validate signup form
function validateSignupForm(form) {
    const password = form.querySelector('#password').value;
    const confirmPassword = form.querySelector('#confirm_password').value;
    const ageVerification = form.querySelector('#age_verification').value;
    const termsAccepted = form.querySelector('input[name="terms_accepted"]').checked;
    const responsibleGambling = form.querySelector('input[name="responsible_gambling"]').checked;
    
    let isValid = true;
    
    // Check password match
    if (password !== confirmPassword) {
        showFieldError('confirm_password', 'Passwords do not match');
        isValid = false;
    }
    
    // Check age verification
    if (ageVerification) {
        const birthDate = new Date(ageVerification);
        const today = new Date();
        const age = today.getFullYear() - birthDate.getFullYear();
        const monthDiff = today.getMonth() - birthDate.getMonth();
        
        if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birthDate.getDate())) {
            age--;
        }
        
        if (age < 18) {
            showFieldError('age_verification', 'You must be 18 or older to register');
            isValid = false;
        }
    }
    
    // Check required checkboxes
    if (!termsAccepted) {
        showError('You must accept the Terms of Service');
        isValid = false;
    }
    
    if (!responsibleGambling) {
        showError('You must acknowledge responsible gambling practices');
        isValid = false;
    }
    
    return isValid;
}

// Password validation and strength meter
function initPasswordValidation() {
    const passwordInputs = document.querySelectorAll('input[type="password"]');
    
    passwordInputs.forEach(input => {
        if (input.id === 'password') {
            input.addEventListener('input', function() {
                updatePasswordStrength(this.value);
            });
        }
        
        input.addEventListener('input', function() {
            if (this.id === 'confirm_password') {
                validatePasswordMatch();
            }
        });
    });
}

// Update password strength indicator
function updatePasswordStrength(password) {
    const strengthBar = document.querySelector('.strength-fill');
    const strengthText = document.querySelector('.strength-text');
    
    if (!strengthBar || !strengthText) return;
    
    let strength = 0;
    let strengthLabel = 'Weak';
    
    // Check password criteria
    if (password.length >= 8) strength++;
    if (/[a-z]/.test(password)) strength++;
    if (/[A-Z]/.test(password)) strength++;
    if (/[0-9]/.test(password)) strength++;
    if (/[^A-Za-z0-9]/.test(password)) strength++;
    
    // Update strength bar
    strengthBar.className = 'strength-fill';
    
    if (strength === 0) {
        strengthLabel = 'Too weak';
    } else if (strength <= 2) {
        strengthBar.classList.add('weak');
        strengthLabel = 'Weak';
    } else if (strength === 3) {
        strengthBar.classList.add('fair');
        strengthLabel = 'Fair';
    } else if (strength === 4) {
        strengthBar.classList.add('good');
        strengthLabel = 'Good';
    } else {
        strengthBar.classList.add('strong');
        strengthLabel = 'Strong';
    }
    
    strengthText.textContent = strengthLabel;
}

// Validate password match
function validatePasswordMatch() {
    const password = document.getElementById('password');
    const confirmPassword = document.getElementById('confirm_password');
    
    if (!password || !confirmPassword) return;
    
    if (confirmPassword.value && password.value !== confirmPassword.value) {
        showFieldError('confirm_password', 'Passwords do not match');
    } else {
        clearFieldError('confirm_password');
    }
}

// Username validation
function initUsernameValidation() {
    const usernameInput = document.getElementById('username');
    if (!usernameInput) return;
    
    let validationTimeout;
    
    usernameInput.addEventListener('input', function() {
        clearTimeout(validationTimeout);
        validationTimeout = setTimeout(() => {
            validateUsername(this.value);
        }, 500);
    });
}

// Validate username availability
async function validateUsername(username) {
    const usernameCheck = document.querySelector('.username-check');
    if (!usernameCheck) return;
    
    if (username.length < 3) {
        usernameCheck.classList.remove('visible');
        return;
    }
    
    try {
        const response = await fetch(`/api.php?route=api/check-username&username=${encodeURIComponent(username)}`);
        const result = await response.json();
        
        if (result.available) {
            usernameCheck.classList.add('visible');
            clearFieldError('username');
        } else {
            usernameCheck.classList.remove('visible');
            showFieldError('username', 'Username is already taken');
        }
    } catch (error) {
        console.error('Username validation error:', error);
        usernameCheck.classList.remove('visible');
    }
}

// Admin toggle functionality
function initAdminToggle() {
    const adminToggle = document.getElementById('adminToggle');
    if (!adminToggle) return;
    
    adminToggle.addEventListener('click', function() {
        const form = document.getElementById('loginForm');
        const isAdminMode = form.classList.contains('admin-mode');
        
        if (isAdminMode) {
            // Switch to regular mode
            form.classList.remove('admin-mode');
            this.innerHTML = '<i class="fas fa-user-shield"></i> Admin Login';
            form.querySelector('.form-header h2').textContent = 'Sign In';
            form.querySelector('.form-header p').textContent = 'Enter your credentials to access your account';
        } else {
            // Switch to admin mode
            form.classList.add('admin-mode');
            this.innerHTML = '<i class="fas fa-user"></i> User Login';
            form.querySelector('.form-header h2').textContent = 'Admin Login';
            form.querySelector('.form-header p').textContent = 'Enter admin credentials to access the dashboard';
        }
    });
}

// Password visibility toggle
function togglePassword(inputId) {
    const input = document.getElementById(inputId);
    const toggle = input.parentNode.querySelector('.password-toggle i');
    
    if (input.type === 'password') {
        input.type = 'text';
        toggle.className = 'fas fa-eye-slash';
    } else {
        input.type = 'password';
        toggle.className = 'fas fa-eye';
    }
}

// Show loading state
function showLoading(form) {
    const submitBtn = form.querySelector('button[type="submit"]');
    submitBtn.classList.add('loading');
    submitBtn.disabled = true;
    
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.classList.add('active');
    }
}

// Hide loading state
function hideLoading(form) {
    const submitBtn = form.querySelector('button[type="submit"]');
    submitBtn.classList.remove('loading');
    submitBtn.disabled = false;
    
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.classList.remove('active');
    }
}

// Show error message
function showError(message) {
    const errorModal = document.getElementById('errorModal');
    const errorMessage = document.getElementById('errorMessage');
    
    if (errorModal && errorMessage) {
        errorMessage.textContent = message;
        showModal('errorModal');
    } else {
        alert(message); // Fallback
    }
}

// Show field-specific error
function showFieldError(fieldId, message) {
    const field = document.getElementById(fieldId);
    if (!field) return;
    
    const wrapper = field.closest('.input-wrapper');
    const formGroup = field.closest('.form-group');
    
    // Remove existing error
    clearFieldError(fieldId);
    
    // Add error class
    wrapper.classList.add('error');
    
    // Create error message
    const errorElement = document.createElement('div');
    errorElement.className = 'error-message';
    errorElement.innerHTML = `<i class="fas fa-exclamation-circle"></i> ${message}`;
    
    formGroup.appendChild(errorElement);
}

// Clear field error
function clearFieldError(fieldId) {
    const field = document.getElementById(fieldId);
    if (!field) return;
    
    const wrapper = field.closest('.input-wrapper');
    const formGroup = field.closest('.form-group');
    const errorMessage = formGroup.querySelector('.error-message');
    
    wrapper.classList.remove('error');
    if (errorMessage) {
        errorMessage.remove();
    }
}

// Modal functions
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

// Redirect to login after successful signup
function redirectToLogin() {
    closeModal('successModal');
    setTimeout(() => {
        window.location.href = 'login.html';
    }, 500);
}

// Social login handlers
function handleGoogleLogin() {
    // Placeholder for Google OAuth integration
    console.log('Google login clicked');
    showError('Social login coming soon!');
}

function handleAppleLogin() {
    // Placeholder for Apple OAuth integration
    console.log('Apple login clicked');
    showError('Social login coming soon!');
}

// Initialize social login buttons
document.addEventListener('DOMContentLoaded', function() {
    const googleBtns = document.querySelectorAll('.btn-google');
    const appleBtns = document.querySelectorAll('.btn-apple');
    
    googleBtns.forEach(btn => {
        btn.addEventListener('click', handleGoogleLogin);
    });
    
    appleBtns.forEach(btn => {
        btn.addEventListener('click', handleAppleLogin);
    });
});

// Form auto-save functionality
function initFormAutoSave() {
    const forms = document.querySelectorAll('form');
    
    forms.forEach(form => {
        const inputs = form.querySelectorAll('input:not([type="password"])');
        
        inputs.forEach(input => {
            // Load saved data
            const savedValue = localStorage.getItem(`form_${form.id}_${input.name}`);
            if (savedValue && input.type !== 'checkbox') {
                input.value = savedValue;
            } else if (savedValue && input.type === 'checkbox') {
                input.checked = savedValue === 'true';
            }
            
            // Save data on change
            input.addEventListener('change', function() {
                if (this.type === 'checkbox') {
                    localStorage.setItem(`form_${form.id}_${this.name}`, this.checked);
                } else {
                    localStorage.setItem(`form_${form.id}_${this.name}`, this.value);
                }
            });
        });
        
        // Clear saved data on successful submission
        form.addEventListener('submit', function() {
            inputs.forEach(input => {
                localStorage.removeItem(`form_${form.id}_${input.name}`);
            });
        });
    });
}

// Initialize auto-save
initFormAutoSave();

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Close modal with Escape key
    if (e.key === 'Escape') {
        const activeModal = document.querySelector('.modal.active');
        if (activeModal) {
            activeModal.classList.remove('active');
            document.body.style.overflow = '';
        }
    }
    
    // Submit form with Ctrl/Cmd + Enter
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        const activeForm = document.activeElement.closest('form');
        if (activeForm) {
            activeForm.requestSubmit();
        }
    }
});

// Handle browser back/forward buttons
window.addEventListener('popstate', function(e) {
    // Close any open modals
    const activeModal = document.querySelector('.modal.active');
    if (activeModal) {
        activeModal.classList.remove('active');
        document.body.style.overflow = '';
    }
});

// Session management
function checkSession() {
    const token = getAuthToken();
    if (!token) {
        console.log('No token found');
        return;
    }
    
    makeAuthenticatedRequest('/api.php?route=api/session')
        .then(response => {
            if (response && response.ok) {
                return response.json();
            } else {
                throw new Error('Session check failed');
            }
        })
        .then(data => {
            if (data.authenticated) {
                // User is already logged in, redirect to dashboard
                if (data.user.is_admin) {
                    window.location.href = 'admin-dashboard.html';
                } else {
                    window.location.href = 'dashboard.html';
                }
            }
        })
        .catch(error => {
            console.log('No active session:', error.message);
            // Clear invalid auth data
            clearAuthData();
        });
}

// Check session on page load (only if we have a token and it's not expired)
if (window.location.pathname.includes('login.html')) {
    const token = getAuthToken();
    if (token && isAuthenticated()) {
        checkSession();
    }
}

// Export functions for global use
window.AuthUtils = {
    togglePassword,
    showModal,
    closeModal,
    redirectToLogin,
    showError,
    showFieldError,
    clearFieldError
};
