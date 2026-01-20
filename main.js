// Main JavaScript for FraudShield AI
document.addEventListener('DOMContentLoaded', function() {
    // Mobile menu toggle
    const menuToggle = document.getElementById('menuToggle');
    const navMenu = document.querySelector('.nav-menu');
    
    if (menuToggle && navMenu) {
        menuToggle.addEventListener('click', () => {
            navMenu.classList.toggle('active');
        });
        
        // Close menu when clicking outside
        document.addEventListener('click', (event) => {
            if (!event.target.closest('.nav-container') && navMenu.classList.contains('active')) {
                navMenu.classList.remove('active');
            }
        });
        
        // Close menu on window resize
        window.addEventListener('resize', () => {
            if (window.innerWidth > 768) {
                navMenu.classList.remove('active');
            }
        });
    }
    
    // Add ripple effect to buttons
    document.querySelectorAll('.btn').forEach(button => {
        button.addEventListener('click', function(e) {
            const ripple = document.createElement('span');
            const rect = this.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            const x = e.clientX - rect.left - size / 2;
            const y = e.clientY - rect.top - size / 2;
            
            ripple.style.cssText = `
                position: absolute;
                border-radius: 50%;
                background: rgba(255, 255, 255, 0.6);
                transform: scale(0);
                animation: ripple 0.6s linear;
                width: ${size}px;
                height: ${size}px;
                top: ${y}px;
                left: ${x}px;
                pointer-events: none;
            `;
            
            this.style.position = 'relative';
            this.style.overflow = 'hidden';
            this.appendChild(ripple);
            
            setTimeout(() => ripple.remove(), 600);
        });
    });
    
    // Add ripple animation style
    if (!document.querySelector('#ripple-style')) {
        const style = document.createElement('style');
        style.id = 'ripple-style';
        style.textContent = `
            @keyframes ripple {
                to {
                    transform: scale(4);
                    opacity: 0;
                }
            }
        `;
        document.head.appendChild(style);
    }
});

// API Utility Functions
async function fetchAPI(endpoint, options = {}) {
    const baseURL = 'http://localhost:5000';
    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
        },
    };
    
    try {
        const response = await fetch(`${baseURL}${endpoint}`, {
            ...defaultOptions,
            ...options
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API Request failed:', error);
        return {
            success: false,
            error: error.message,
            details: 'Please make sure the backend server is running on http://localhost:5000'
        };
    }
}

// Check backend connection
async function checkBackendConnection() {
    try {
        const response = await fetchAPI('/api/health');
        return response.status === 'healthy';
    } catch (error) {
        return false;
    }
}

// Export for use in dashboard.js
if (typeof module === 'object' && module.exports) {
    module.exports = { fetchAPI, checkBackendConnection };
}