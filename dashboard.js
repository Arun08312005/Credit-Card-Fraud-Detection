// Dashboard JavaScript
document.addEventListener('DOMContentLoaded', function() {
    // Initialize dashboard
    initDashboard();
});

async function initDashboard() {
    // Initialize components
    initModelControls();
    initTransactionTesting();
    initActivityList();
    updateConnectionStatus();
    
    // Load initial data
    await loadModelStatus();
    await loadRecentActivity();
}

// Model Controls
function initModelControls() {
    // Train Model button
    const trainBtn = document.getElementById('trainModel');
    if (trainBtn) {
        trainBtn.addEventListener('click', startModelTraining);
    }
    
    // Load Model button
    const loadBtn = document.getElementById('loadModel');
    if (loadBtn) {
        loadBtn.addEventListener('click', loadModel);
    }
    
    // Sensitivity slider
    const sensitivitySlider = document.getElementById('sensitivity');
    if (sensitivitySlider) {
        sensitivitySlider.addEventListener('input', function() {
            console.log('Sensitivity changed to:', this.value);
        });
    }
}

// Transaction Testing
function initTransactionTesting() {
    // Test Transaction button
    const testBtn = document.getElementById('testTransaction');
    if (testBtn) {
        testBtn.addEventListener('click', testTransaction);
    }
    
    // Random Test button
    const randomBtn = document.getElementById('randomTest');
    if (randomBtn) {
        randomBtn.addEventListener('click', testRandomTransaction);
    }
}

// Activity List
function initActivityList() {
    const refreshBtn = document.getElementById('refreshActivity');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', loadRecentActivity);
    }
}

// Update connection status
async function updateConnectionStatus() {
    const statusEl = document.getElementById('connectionStatus');
    if (!statusEl) return;
    
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        
        if (data.status === 'healthy') {
            statusEl.innerHTML = '<i class="fas fa-check-circle"></i> Connected';
            statusEl.className = 'status-badge success';
        } else {
            statusEl.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Disconnected';
            statusEl.className = 'status-badge warning';
        }
    } catch (error) {
        statusEl.innerHTML = '<i class="fas fa-times-circle"></i> No Connection';
        statusEl.className = 'status-badge warning';
    }
}

// Load model status
async function loadModelStatus() {
    const statusEl = document.getElementById('modelStatus');
    if (!statusEl) return;
    
    try {
        const response = await fetch('/api/model/status');
        const data = await response.json();
        
        if (data.is_trained) {
            statusEl.innerHTML = '<p><i class="fas fa-check-circle" style="color: #4ECDC4;"></i> Model is trained and ready</p>';
        } else if (data.has_model_files) {
            statusEl.innerHTML = '<p><i class="fas fa-exclamation-triangle" style="color: #FFD166;"></i> Model files exist but not loaded</p>';
        } else {
            statusEl.innerHTML = '<p><i class="fas fa-times-circle" style="color: #EF476F;"></i> No model found. Click "Train Model"</p>';
        }
    } catch (error) {
        statusEl.innerHTML = '<p><i class="fas fa-exclamation-triangle" style="color: #EF476F;"></i> Failed to load model status</p>';
    }
}

// Start model training
async function startModelTraining() {
    const trainingSection = document.getElementById('trainingSection');
    const trainBtn = document.getElementById('trainModel');
    
    if (trainingSection) trainingSection.style.display = 'block';
    if (trainBtn) trainBtn.disabled = true;
    
    // Update progress
    updateTrainingProgress(10, 'Starting training process...');
    
    try {
        const response = await fetch('/api/model/train', { method: 'POST' });
        const data = await response.json();
        
        if (data.success) {
            // Simulate training progress
            simulateTrainingProgress();
        } else {
            throw new Error(data.error || 'Training failed');
        }
    } catch (error) {
        updateTrainingProgress(0, `Error: ${error.message}`);
        if (trainBtn) trainBtn.disabled = false;
    }
}

// Simulate training progress
function simulateTrainingProgress() {
    let progress = 10;
    const messages = [
        'Loading dataset...',
        'Preprocessing data...',
        'Training Random Forest model...',
        'Calculating metrics...',
        'Saving model...',
        'Training complete!'
    ];
    
    const interval = setInterval(() => {
        progress += 15;
        if (progress > 100) progress = 100;
        
        const messageIndex = Math.floor((progress - 10) / 15);
        updateTrainingProgress(progress, messages[messageIndex] || 'Finishing up...');
        
        if (progress >= 100) {
            clearInterval(interval);
            
            // Re-enable button after delay
            setTimeout(() => {
                const trainBtn = document.getElementById('trainModel');
                if (trainBtn) trainBtn.disabled = false;
                
                // Reload model status
                loadModelStatus();
            }, 2000);
        }
    }, 1000);
}

// Update training progress
function updateTrainingProgress(progress, message) {
    const progressEl = document.getElementById('trainingProgress');
    const barEl = document.getElementById('trainingBar');
    const messageEl = document.getElementById('trainingMessage');
    
    if (progressEl) progressEl.textContent = `${progress}%`;
    if (barEl) barEl.style.width = `${progress}%`;
    if (messageEl) messageEl.textContent = message;
}

// Load model
async function loadModel() {
    const loadBtn = document.getElementById('loadModel');
    if (loadBtn) loadBtn.disabled = true;
    
    try {
        const response = await fetch('/api/model/load', { method: 'POST' });
        const data = await response.json();
        
        if (data.success) {
            alert('Model loaded successfully!');
            loadModelStatus();
        } else {
            alert(`Error: ${data.error || 'Failed to load model'}`);
        }
    } catch (error) {
        alert(`Error: ${error.message}`);
    } finally {
        if (loadBtn) loadBtn.disabled = false;
    }
}

// Test transaction
async function testTransaction() {
    const amountInput = document.getElementById('testAmount');
    const amount = amountInput ? parseFloat(amountInput.value) : 250;
    
    if (!amount || amount <= 0) {
        alert('Please enter a valid amount');
        return;
    }
    
    // Show loading
    showResultLoading();
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ amount: amount })
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayResult(data.prediction);
            updateQuickStats(data.prediction);
            addToActivityList(data.prediction);
        } else {
            showResultError(data.error || 'Prediction failed');
        }
    } catch (error) {
        showResultError('Connection error. Please check if backend is running.');
    }
}

// Test random transaction
async function testRandomTransaction() {
    // Show loading
    showResultLoading();
    
    try {
        const response = await fetch('/api/demo/predict');
        const data = await response.json();
        
        if (data.success) {
            displayResult(data.prediction);
            updateQuickStats(data.prediction);
            addToActivityList(data.prediction);
        } else {
            showResultError(data.error || 'Demo failed');
        }
    } catch (error) {
        // Fallback to offline demo
        showOfflineDemoResult();
    }
}

// Show result loading
function showResultLoading() {
    const resultContent = document.getElementById('resultContent');
    const resultStatus = document.getElementById('resultStatus');
    
    if (resultContent) {
        resultContent.innerHTML = `
            <div class="loading">
                <i class="fas fa-spinner fa-spin"></i>
                <p>Analyzing transaction...</p>
            </div>
        `;
    }
    
    if (resultStatus) {
        resultStatus.textContent = 'Analyzing...';
        resultStatus.className = 'status-badge warning';
    }
}

// Display result
function displayResult(prediction) {
    const resultContent = document.getElementById('resultContent');
    const resultStatus = document.getElementById('resultStatus');
    
    if (!resultContent || !resultStatus) return;
    
    const isFraud = prediction.is_fraud;
    const riskScore = prediction.risk_score;
    const probability = (prediction.probability * 100).toFixed(1);
    
    // Determine status
    let statusClass = 'success';
    let statusText = 'SAFE';
    let statusColor = '#4ECDC4';
    
    if (riskScore >= 80) {
        statusClass = 'warning';
        statusText = 'FRAUD';
        statusColor = '#EF476F';
    } else if (riskScore >= 60) {
        statusClass = 'warning';
        statusText = 'SUSPICIOUS';
        statusColor = '#FFD166';
    }
    
    // Update status badge
    resultStatus.textContent = statusText;
    resultStatus.className = `status-badge ${statusClass}`;
    
    // Create result HTML
    const transaction = prediction.transaction || prediction.transaction_details || {};
    
    resultContent.innerHTML = `
        <div class="result-metrics">
            <div class="metric">
                <div class="metric-label">Risk Score</div>
                <div class="metric-value" style="color: ${statusColor}">${riskScore}/100</div>
                <div class="metric-bar">
                    <div class="metric-fill" style="width: ${riskScore}%; background: ${statusColor}"></div>
                </div>
            </div>
            <div class="metric">
                <div class="metric-label">Fraud Probability</div>
                <div class="metric-value">${probability}%</div>
            </div>
            <div class="metric">
                <div class="metric-label">Prediction</div>
                <div class="metric-value">${isFraud ? 'FRAUD' : 'LEGITIMATE'}</div>
            </div>
        </div>
        
        <div class="transaction-details">
            <h4><i class="fas fa-receipt"></i> Transaction Details</h4>
            ${transaction.id ? `<div class="detail-item"><span class="detail-label">ID:</span><span class="detail-value">${transaction.id}</span></div>` : ''}
            ${transaction.amount ? `<div class="detail-item"><span class="detail-label">Amount:</span><span class="detail-value">${transaction.amount}</span></div>` : ''}
            ${transaction.merchant ? `<div class="detail-item"><span class="detail-label">Merchant:</span><span class="detail-value">${transaction.merchant}</span></div>` : ''}
            ${transaction.location ? `<div class="detail-item"><span class="detail-label">Location:</span><span class="detail-value">${transaction.location}</span></div>` : ''}
            ${transaction.time ? `<div class="detail-item"><span class="detail-label">Time:</span><span class="detail-value">${transaction.time}</span></div>` : ''}
        </div>
        
        ${prediction.recommendation ? `
        <div class="result-recommendation">
            <h4><i class="fas fa-lightbulb"></i> Recommendation</h4>
            <p>${prediction.recommendation.message}</p>
            <div class="detail-item" style="margin-top: 1rem;">
                <span class="detail-label">Action:</span>
                <span class="detail-value" style="color: ${prediction.recommendation.color || statusColor}">${prediction.recommendation.action}</span>
            </div>
        </div>
        ` : ''}
    `;
}

// Show result error
function showResultError(message) {
    const resultContent = document.getElementById('resultContent');
    const resultStatus = document.getElementById('resultStatus');
    
    if (resultContent) {
        resultContent.innerHTML = `
            <div class="error">
                <i class="fas fa-exclamation-triangle"></i>
                <p>${message}</p>
            </div>
        `;
    }
    
    if (resultStatus) {
        resultStatus.textContent = 'Error';
        resultStatus.className = 'status-badge warning';
    }
}

// Show offline demo result
function showOfflineDemoResult() {
    // Generate random offline result
    const isFraud = Math.random() > 0.7;
    const riskScore = isFraud ? Math.floor(Math.random() * 30) + 70 : Math.floor(Math.random() * 40);
    const probability = (Math.random() * 20 + 80).toFixed(1);
    
    const prediction = {
        is_fraud: isFraud,
        risk_score: riskScore,
        probability: probability / 100,
        transaction: {
            id: `DEMO${Math.floor(Math.random() * 9000) + 1000}`,
            amount: `$${Math.floor(Math.random() * 5000) + 100}`,
            merchant: ['Amazon', 'Walmart', 'Uber', 'Netflix', 'Unknown'][Math.floor(Math.random() * 5)],
            location: ['New York', 'London', 'Tokyo', 'Moscow', 'Online'][Math.floor(Math.random() * 5)],
            time: `${String(Math.floor(Math.random() * 24)).padStart(2, '0')}:${String(Math.floor(Math.random() * 60)).padStart(2, '0')}`
        },
        recommendation: {
            message: isFraud ? 'High fraud risk detected in offline demo mode.' : 'Transaction appears legitimate in offline demo mode.',
            action: isFraud ? 'REVIEW' : 'APPROVE',
            color: isFraud ? '#EF476F' : '#4ECDC4'
        }
    };
    
    displayResult(prediction);
}

// Update quick stats
function updateQuickStats(prediction) {
    const quickStats = document.getElementById('quickStats');
    if (!quickStats) return;
    
    // In a real app, you'd track statistics
    const stats = {
        total: parseInt(quickStats.dataset.total || '0') + 1,
        fraud: parseInt(quickStats.dataset.fraud || '0') + (prediction.is_fraud ? 1 : 0),
        avgRisk: parseFloat(quickStats.dataset.avgRisk || '0')
    };
    
    stats.avgRisk = ((stats.avgRisk * (stats.total - 1)) + prediction.risk_score) / stats.total;
    
    // Save stats
    quickStats.dataset.total = stats.total;
    quickStats.dataset.fraud = stats.fraud;
    quickStats.dataset.avgRisk = stats.avgRisk.toFixed(1);
    
    // Update display
    quickStats.innerHTML = `
        <div class="detail-item">
            <span class="detail-label">Total Tests:</span>
            <span class="detail-value">${stats.total}</span>
        </div>
        <div class="detail-item">
            <span class="detail-label">Fraud Detected:</span>
            <span class="detail-value">${stats.fraud}</span>
        </div>
        <div class="detail-item">
            <span class="detail-label">Avg Risk Score:</span>
            <span class="detail-value">${stats.avgRisk.toFixed(1)}/100</span>
        </div>
    `;
}

// Add to activity list
function addToActivityList(prediction) {
    const activityList = document.getElementById('activityList');
    if (!activityList) return;
    
    const transaction = prediction.transaction || prediction.transaction_details || {};
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    
    const activityItem = document.createElement('div');
    activityItem.className = 'detail-item fade-in';
    activityItem.innerHTML = `
        <span class="detail-label">
            <i class="fas ${prediction.is_fraud ? 'fa-exclamation-triangle' : 'fa-check-circle'}" 
               style="color: ${prediction.is_fraud ? '#EF476F' : '#4ECDC4'}; margin-right: 0.5rem;"></i>
            ${transaction.id || 'Transaction'}
        </span>
        <span class="detail-value">
            ${prediction.is_fraud ? 'FRAUD' : 'SAFE'} • ${time}
        </span>
    `;
    
    // Add to top of list
    if (activityList.firstChild) {
        activityList.insertBefore(activityItem, activityList.firstChild);
    } else {
        activityList.appendChild(activityItem);
    }
    
    // Limit to 10 items
    const items = activityList.querySelectorAll('.detail-item');
    if (items.length > 10) {
        activityList.removeChild(items[items.length - 1]);
    }
}

// Load recent activity
async function loadRecentActivity() {
    const activityList = document.getElementById('activityList');
    if (!activityList) return;
    
    try {
        const response = await fetch('/api/transactions/recent');
        const data = await response.json();
        
        if (data.success) {
            activityList.innerHTML = '';
            
            data.transactions.forEach(transaction => {
                const activityItem = document.createElement('div');
                activityItem.className = 'detail-item';
                activityItem.innerHTML = `
                    <span class="detail-label">
                        <i class="fas ${transaction.status === 'fraud' ? 'fa-exclamation-triangle' : 'fa-check-circle'}" 
                           style="color: ${transaction.status === 'fraud' ? '#EF476F' : '#4ECDC4'}; margin-right: 0.5rem;"></i>
                        ${transaction.id}
                    </span>
                    <span class="detail-value">
                        ${transaction.status.toUpperCase()} • ${transaction.time}
                    </span>
                `;
                activityList.appendChild(activityItem);
            });
        }
    } catch (error) {
        // Use sample data if API fails
        activityList.innerHTML = `
            <div class="detail-item">
                <span class="detail-label"><i class="fas fa-check-circle" style="color: #4ECDC4; margin-right: 0.5rem;"></i> TX1001</span>
                <span class="detail-value">SAFE • 14:30</span>
            </div>
            <div class="detail-item">
                <span class="detail-label"><i class="fas fa-exclamation-triangle" style="color: #EF476F; margin-right: 0.5rem;"></i> TX1002</span>
                <span class="detail-value">FRAUD • 14:31</span>
            </div>
            <div class="detail-item">
                <span class="detail-label"><i class="fas fa-check-circle" style="color: #4ECDC4; margin-right: 0.5rem;"></i> TX1003</span>
                <span class="detail-value">SAFE • 14:32</span>
            </div>
        `;
    }
}