// Main JavaScript for GoonSteen Website

document.addEventListener('DOMContentLoaded', function() {
    // Initialize all components
    initNavigation();
    initScrollAnimations();
    initSmoothScrolling();
    initMobileMenu();
    initScrollToTop();
    initFormValidation();
});

// Navigation functionality
function initNavigation() {
    const navbar = document.querySelector('.navbar');
    const navLinks = document.querySelectorAll('.nav-link');
    
    // Handle scroll effect on navbar
    window.addEventListener('scroll', function() {
        if (window.scrollY > 100) {
            navbar.style.background = 'rgba(255, 255, 255, 0.98)';
            navbar.style.boxShadow = '0 4px 6px -1px rgba(0, 0, 0, 0.1)';
        } else {
            navbar.style.background = 'rgba(255, 255, 255, 0.95)';
            navbar.style.boxShadow = 'none';
        }
    });
    
    // Handle active nav link
    window.addEventListener('scroll', function() {
        const sections = document.querySelectorAll('section[id]');
        const scrollPos = window.scrollY + 100;
        
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.offsetHeight;
            const sectionId = section.getAttribute('id');
            
            if (scrollPos >= sectionTop && scrollPos < sectionTop + sectionHeight) {
                navLinks.forEach(link => link.classList.remove('active'));
                const activeLink = document.querySelector(`.nav-link[href="#${sectionId}"]`);
                if (activeLink) {
                    activeLink.classList.add('active');
                }
            }
        });
    });
}

// Smooth scrolling for navigation links
function initSmoothScrolling() {
    const links = document.querySelectorAll('a[href^="#"]');
    
    links.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href').substring(1);
            const targetSection = document.getElementById(targetId);
            
            if (targetSection) {
                const offsetTop = targetSection.offsetTop - 80;
                
                window.scrollTo({
                    top: offsetTop,
                    behavior: 'smooth'
                });
            }
        });
    });
}

// Mobile menu functionality
function initMobileMenu() {
    const hamburger = document.querySelector('.hamburger');
    const navMenu = document.querySelector('.nav-menu');
    const navButtons = document.querySelector('.nav-buttons');
    
    if (hamburger) {
        hamburger.addEventListener('click', function() {
            hamburger.classList.toggle('active');
            
            // Create mobile menu if it doesn't exist
            let mobileMenu = document.querySelector('.mobile-menu');
            if (!mobileMenu) {
                mobileMenu = createMobileMenu();
                document.body.appendChild(mobileMenu);
            }
            
            mobileMenu.classList.toggle('active');
            document.body.classList.toggle('menu-open');
        });
    }
}

function createMobileMenu() {
    const mobileMenu = document.createElement('div');
    mobileMenu.className = 'mobile-menu';
    
    const menuContent = `
        <div class="mobile-menu-content">
            <a href="#home" class="mobile-nav-link">Home</a>
            <a href="#features" class="mobile-nav-link">Features</a>
            <a href="#about" class="mobile-nav-link">About</a>
            <a href="#contact" class="mobile-nav-link">Contact</a>
            <div class="mobile-buttons">
                <a href="login.html" class="btn btn-outline">Login</a>
                <a href="signup.html" class="btn btn-primary">Sign Up</a>
            </div>
        </div>
    `;
    
    mobileMenu.innerHTML = menuContent;
    
    // Add mobile menu styles
    const style = document.createElement('style');
    style.textContent = `
        .mobile-menu {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100vh;
            background: rgba(0, 0, 0, 0.5);
            z-index: 9999;
            opacity: 0;
            visibility: hidden;
            transition: all 0.3s ease;
        }
        
        .mobile-menu.active {
            opacity: 1;
            visibility: visible;
        }
        
        .mobile-menu-content {
            position: absolute;
            top: 0;
            right: 0;
            width: 280px;
            height: 100vh;
            background: white;
            padding: 2rem;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            transform: translateX(100%);
            transition: transform 0.3s ease;
        }
        
        .mobile-menu.active .mobile-menu-content {
            transform: translateX(0);
        }
        
        .mobile-nav-link {
            color: var(--text-primary);
            text-decoration: none;
            font-weight: 500;
            font-size: 1.125rem;
            padding: 0.75rem 0;
            border-bottom: 1px solid var(--border-color);
        }
        
        .mobile-buttons {
            display: flex;
            flex-direction: column;
            gap: 1rem;
            margin-top: 2rem;
        }
        
        body.menu-open {
            overflow: hidden;
        }
        
        .hamburger.active span:nth-child(1) {
            transform: rotate(45deg) translate(5px, 5px);
        }
        
        .hamburger.active span:nth-child(2) {
            opacity: 0;
        }
        
        .hamburger.active span:nth-child(3) {
            transform: rotate(-45deg) translate(7px, -6px);
        }
    `;
    
    document.head.appendChild(style);
    
    // Add click handlers for mobile nav links
    const mobileNavLinks = mobileMenu.querySelectorAll('.mobile-nav-link');
    mobileNavLinks.forEach(link => {
        link.addEventListener('click', function() {
            mobileMenu.classList.remove('active');
            document.body.classList.remove('menu-open');
            document.querySelector('.hamburger').classList.remove('active');
        });
    });
    
    // Close menu when clicking outside
    mobileMenu.addEventListener('click', function(e) {
        if (e.target === mobileMenu) {
            mobileMenu.classList.remove('active');
            document.body.classList.remove('menu-open');
            document.querySelector('.hamburger').classList.remove('active');
        }
    });
    
    return mobileMenu;
}

// Scroll animations
function initScrollAnimations() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-on-scroll');
            }
        });
    }, observerOptions);
    
    // Observe elements for animation
    const animateElements = document.querySelectorAll('.feature-card, .about-text, .stats-dashboard, .contact-item, .cta-section');
    animateElements.forEach(el => {
        observer.observe(el);
    });
    
    // Add staggered animation delays
    const featureCards = document.querySelectorAll('.feature-card');
    featureCards.forEach((card, index) => {
        card.style.animationDelay = `${index * 0.1}s`;
    });
}

// Scroll to top functionality
function initScrollToTop() {
    // Create scroll to top button
    const scrollTopBtn = document.createElement('button');
    scrollTopBtn.innerHTML = '<i class="fas fa-arrow-up"></i>';
    scrollTopBtn.className = 'scroll-top-btn';
    scrollTopBtn.setAttribute('aria-label', 'Scroll to top');
    
    // Add styles
    const style = document.createElement('style');
    style.textContent = `
        .scroll-top-btn {
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            width: 3rem;
            height: 3rem;
            background: var(--gradient-primary);
            color: white;
            border: none;
            border-radius: 50%;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1rem;
            box-shadow: var(--shadow-lg);
            opacity: 0;
            visibility: hidden;
            transform: translateY(20px);
            transition: all 0.3s ease;
            z-index: 1000;
        }
        
        .scroll-top-btn.visible {
            opacity: 1;
            visibility: visible;
            transform: translateY(0);
        }
        
        .scroll-top-btn:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-xl);
        }
    `;
    
    document.head.appendChild(style);
    document.body.appendChild(scrollTopBtn);
    
    // Show/hide button based on scroll position
    window.addEventListener('scroll', function() {
        if (window.scrollY > 500) {
            scrollTopBtn.classList.add('visible');
        } else {
            scrollTopBtn.classList.remove('visible');
        }
    });
    
    // Scroll to top when clicked
    scrollTopBtn.addEventListener('click', function() {
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    });
}

// Form validation (for future forms)
function initFormValidation() {
    const forms = document.querySelectorAll('form');
    
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            const inputs = form.querySelectorAll('input[required], select[required], textarea[required]');
            let isValid = true;
            
            inputs.forEach(input => {
                if (!validateInput(input)) {
                    isValid = false;
                }
            });
            
            if (!isValid) {
                e.preventDefault();
            }
        });
        
        // Real-time validation
        const inputs = form.querySelectorAll('input, select, textarea');
        inputs.forEach(input => {
            input.addEventListener('blur', function() {
                validateInput(this);
            });
            
            input.addEventListener('input', function() {
                if (this.classList.contains('error')) {
                    validateInput(this);
                }
            });
        });
    });
}

function validateInput(input) {
    const value = input.value.trim();
    const type = input.type;
    let isValid = true;
    let errorMessage = '';
    
    // Remove existing error
    clearInputError(input);
    
    // Check required fields
    if (input.hasAttribute('required') && !value) {
        isValid = false;
        errorMessage = 'This field is required';
    }
    
    // Email validation
    else if (type === 'email' && value) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (!emailRegex.test(value)) {
            isValid = false;
            errorMessage = 'Please enter a valid email address';
        }
    }
    
    // Password validation
    else if (type === 'password' && value) {
        if (value.length < 8) {
            isValid = false;
            errorMessage = 'Password must be at least 8 characters long';
        }
    }
    
    // Confirm password validation
    else if (input.name === 'confirm_password' && value) {
        const passwordField = input.form.querySelector('input[name="password"]');
        if (passwordField && value !== passwordField.value) {
            isValid = false;
            errorMessage = 'Passwords do not match';
        }
    }
    
    if (!isValid) {
        showInputError(input, errorMessage);
    }
    
    return isValid;
}

function showInputError(input, message) {
    input.classList.add('error');
    
    // Create error message element
    const errorElement = document.createElement('div');
    errorElement.className = 'input-error';
    errorElement.textContent = message;
    
    // Insert after input
    input.parentNode.insertBefore(errorElement, input.nextSibling);
    
    // Add error styles if not already added
    if (!document.querySelector('#input-error-styles')) {
        const style = document.createElement('style');
        style.id = 'input-error-styles';
        style.textContent = `
            .error {
                border-color: #ef4444 !important;
                box-shadow: 0 0 0 3px rgba(239, 68, 68, 0.1) !important;
            }
            
            .input-error {
                color: #ef4444;
                font-size: 0.875rem;
                margin-top: 0.25rem;
                margin-bottom: 0.5rem;
            }
        `;
        document.head.appendChild(style);
    }
}

function clearInputError(input) {
    input.classList.remove('error');
    const errorElement = input.parentNode.querySelector('.input-error');
    if (errorElement) {
        errorElement.remove();
    }
}

// Utility functions
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    }
}

// Performance optimizations
const optimizedScrollHandler = throttle(function() {
    // Handle scroll events efficiently
}, 16);

window.addEventListener('scroll', optimizedScrollHandler);

// Preload critical resources
function preloadResources() {
    const criticalImages = [
        // Add any critical images here
    ];
    
    criticalImages.forEach(src => {
        const link = document.createElement('link');
        link.rel = 'preload';
        link.as = 'image';
        link.href = src;
        document.head.appendChild(link);
    });
}

// Initialize preloading
preloadResources();

// Loading state management
window.addEventListener('load', function() {
    document.body.classList.add('loaded');
    
    // Add loaded styles
    const style = document.createElement('style');
    style.textContent = `
        body:not(.loaded) {
            overflow: hidden;
        }
        
        body:not(.loaded)::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: white;
            z-index: 10000;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        body:not(.loaded)::after {
            content: '';
            width: 2rem;
            height: 2rem;
            border: 2px solid var(--border-color);
            border-top-color: var(--primary-color);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            z-index: 10001;
        }
        
        @keyframes spin {
            to { transform: translate(-50%, -50%) rotate(360deg); }
        }
        
        body.loaded::before,
        body.loaded::after {
            display: none;
        }
    `;
    document.head.appendChild(style);
});

// Error handling
window.addEventListener('error', function(e) {
    console.error('JavaScript error:', e.error);
    // You can add error reporting here
});

// Analytics placeholder
function trackEvent(eventName, properties = {}) {
    // Placeholder for analytics tracking
    console.log('Track event:', eventName, properties);
}

// Export functions for use in other files
window.GoonSteen = {
    trackEvent,
    validateInput,
    showInputError,
    clearInputError
};
