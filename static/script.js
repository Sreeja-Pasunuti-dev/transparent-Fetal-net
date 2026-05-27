// script.js

let probChart, shapChart, progressChart;
let currentState = null;

// Format feature names
const formatFeatureName = (name) => {
    return name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
};

function updateMetricCards(state) {
    document.getElementById('metric-iteration').innerText = state.iteration;
    document.getElementById('metric-initial').innerText = state.initialTrainingSize.toLocaleString();
    document.getElementById('metric-queried').innerText = state.queriedSamples;
    document.getElementById('metric-trainsize').innerText = state.trainSize.toLocaleString();
    document.getElementById('metric-pool').innerText = state.remainingPool.toLocaleString();
    document.getElementById('metric-accuracy').innerText = state.accuracy.toFixed(1) + '%';
}

function renderTable(sampleFeatures, featureNames) {
    const tbody = document.getElementById('feature-table-body');
    tbody.innerHTML = '';
    
    featureNames.forEach((name, idx) => {
        let val = sampleFeatures[idx];
        // Highlight potentially concerning values for effect based on name
        let rowClass = '';
        if (name.includes('abnormal') && val > 40) rowClass = 'table-warning';
        
        tbody.innerHTML += `
            <tr class="${rowClass}">
                <td class="ps-4">${formatFeatureName(name)}</td>
                <td class="text-end pe-4 fw-medium font-monospace">${val.toFixed(3)}</td>
            </tr>
        `;
    });
}

function initCharts() {
    Chart.defaults.font.family = "'Inter', sans-serif";
    Chart.defaults.color = '#64748b';

    // Prob chart
    const ctxProb = document.getElementById('probChart').getContext('2d');
    probChart = new Chart(ctxProb, {
        type: 'bar',
        data: {
            labels: ['Normal', 'Suspect', 'Pathol.'],
            datasets: [{
                label: 'Probability',
                data: [0, 0, 0],
                backgroundColor: ['#10b981', '#f59e0b', '#ef4444'],
                borderRadius: 6,
                borderSkipped: false,
                barThickness: 50
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: { 
                y: { 
                    beginAtZero: true, 
                    max: 1,
                    grid: { color: '#f1f5f9' }
                },
                x: {
                    grid: { display: false }
                }
            },
            plugins: { 
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return 'Probability: ' + (context.raw * 100).toFixed(1) + '%';
                        }
                    }
                }
            }
        }
    });

    // SHAP chart
    const ctxShap = document.getElementById('shapChart').getContext('2d');
    shapChart = new Chart(ctxShap, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'SHAP Value (Impact)',
                data: [],
                backgroundColor: (ctx) => {
                    const val = ctx.raw || 0;
                    return val > 0 ? '#ef4444' : '#10b981';
                },
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { 
                    grid: { color: '#f1f5f9' },
                    title: { display: true, text: 'SHAP Value (Impact on Prediction)' }
                },
                y: { grid: { display: false } }
            },
            plugins: { legend: { display: false } }
        }
    });

    // Progress chart
    const ctxProg = document.getElementById('progressChart').getContext('2d');
    progressChart = new Chart(ctxProg, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Model Accuracy (%)',
                data: [],
                borderColor: '#6366f1',
                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                fill: true,
                tension: 0.4,
                borderWidth: 3,
                pointRadius: 4,
                pointBackgroundColor: '#ffffff',
                pointBorderColor: '#6366f1',
                pointBorderWidth: 2,
                pointHoverRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index',
            },
            scales: { 
                y: { 
                    min: 60, 
                    max: 100,
                    grid: { color: '#f1f5f9' },
                    title: { display: true, text: 'Accuracy (%)' }
                },
                x: { 
                    grid: { display: false },
                    title: { display: true, text: 'Active Learning Iteration' } 
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

function updateShapTextAndList(shapValues, featureNames) {
    const shapList = document.getElementById('shap-list');
    shapList.innerHTML = '';
    
    // Create sorted array of objects
    let combined = featureNames.map((l, i) => ({ label: formatFeatureName(l), value: shapValues[i] }));
    // Sort by absolute magnitude
    combined.sort((a,b) => Math.abs(b.value) - Math.abs(a.value));
    
    // Update SHAP Chart with Top 5
    const top5 = combined.slice(0, 5);
    shapChart.data.labels = top5.map(x => x.label);
    shapChart.data.datasets[0].data = top5.map(x => x.value);
    shapChart.update();
    
    // Update List with Top 3
    for(let i=0; i<3; i++) {
        const item = combined[i];
        if (!item) continue;
        const isPos = item.value > 0;
        const colorClass = isPos ? 'bg-danger' : 'bg-success';
        const sign = isPos ? '+' : '';
        const direction = isPos ? 'Pathological/Suspect' : 'Normal';
        
        shapList.innerHTML += `
            <li class="list-group-item px-0 d-flex justify-content-between align-items-center bg-transparent border-light">
                <span class="text-dark fw-medium">${item.label}</span>
                <span class="badge ${colorClass} rounded-pill shadow-sm">${sign}${item.value.toFixed(3)} (${direction})</span>
            </li>
        `;
    }

    // Update conversational text
    const textDiv = document.getElementById('xai-text');
    if (combined.length >= 2) {
        const topFeature = combined[0];
        const secondFeature = combined[1];
        
        let topPush = topFeature.value > 0 ? "Pathological/Suspect classes" : "the Normal class";
        let secondPush = secondFeature.value > 0 ? "Pathological/Suspect classes" : "the Normal class";
        
        const templates = [
            `The model is uncertain because <strong>${topFeature.label.toLowerCase()}</strong> is strongly pushing the prediction toward ${topPush}, while <strong>${secondFeature.label.toLowerCase()}</strong> is pulling it toward ${secondPush}.`,
            `There is conflicting evidence: <strong>${topFeature.label.toLowerCase()}</strong> suggests ${topPush}, but this is counteracted by <strong>${secondFeature.label.toLowerCase()}</strong> which indicates ${secondPush}.`,
            `Uncertainty arises largely due to <strong>${topFeature.label.toLowerCase()}</strong> (supporting ${topPush}), which contradicts the impact of <strong>${secondFeature.label.toLowerCase()}</strong>.`
        ];
        
        textDiv.innerHTML = templates[Math.floor(Math.random() * templates.length)];
    }
}

function updateDashboardData(state) {
    currentState = state;
    
    // Update metrics
    updateMetricCards(state);
    
    // Update Table
    renderTable(state.sample.features, state.sample.featureNames);
    
    // Update Uncertainty indicator
    const probs = state.probabilities;
    const maxP = Math.max(...probs);
    const unc = 1 - maxP;
    
    document.getElementById('uncertainty-score').innerText = unc.toFixed(2);
    const uBar = document.getElementById('uncertainty-bar');
    const uPercent = (unc * 100).toFixed(0) + '%';
    uBar.style.width = uPercent;
    uBar.innerText = uPercent;
    
    const alertDiv = document.getElementById('uncertainty-alert');
    if (unc > 0.4) {
        alertDiv.style.visibility = 'visible';
        uBar.className = 'progress-bar bg-warning progress-bar-striped progress-bar-animated text-dark fw-bold';
    } else {
        alertDiv.style.visibility = 'hidden';
        uBar.className = 'progress-bar bg-info progress-bar-striped progress-bar-animated text-dark fw-bold';
    }
    
    // Update Probability Chart
    probChart.data.datasets[0].data = probs;
    probChart.update();
    
    // Update SHAP Chart & Text
    updateShapTextAndList(state.shap.values, state.sample.featureNames);
    
    // Update Progress Chart
    progressChart.data.labels = Array.from({length: state.progressHistory.length}, (_, i) => i);
    progressChart.data.datasets[0].data = state.progressHistory;
    progressChart.update();
    
    // Reset Form
    document.getElementById('label-select').value = '';
}

async function fetchState() {
    try {
        const response = await fetch('/api/state');
        if (!response.ok) {
            console.error("No more samples or error loading state.");
            return;
        }
        const state = await response.json();
        updateDashboardData(state);
    } catch (error) {
        console.error("Error fetching state:", error);
    }
}

async function submitLabel(label) {
    const btn = document.getElementById('submit-btn');
    const originalContent = btn.innerHTML;
    btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span> Training...';
    btn.disabled = true;

    try {
        const response = await fetch('/api/submit', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ label: label })
        });
        
        if (response.ok) {
            // Re-fetch state to get next sample and new metrics
            await fetchState();
        } else {
            console.error("Failed to submit label");
        }
    } catch (error) {
        console.error("Error submitting label:", error);
    } finally {
        btn.innerHTML = originalContent;
        btn.disabled = false;
    }
}

// Initialization on load
document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    fetchState(); // Boot up initial data
    
    // Form submission listener
    document.getElementById('feedback-form').addEventListener('submit', (e) => {
        e.preventDefault();
        const lbl = document.getElementById('label-select').value;
        if(!lbl) return;
        submitLabel(lbl);
    });
});
