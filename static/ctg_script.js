document.addEventListener('DOMContentLoaded', () => {
    // Form handling
    const form = document.getElementById('ctg-form');
    const resultSection = document.getElementById('result-section');
    const analyzeBtn = document.getElementById('analyze-btn');
    
    // Status Elements
    const statusIndicator = document.querySelector('.status-indicator');
    const fetalHealthStatus = document.getElementById('fetal-health-status');
    const confidenceScore = document.getElementById('confidence-score');
    const riskLevel = document.getElementById('risk-level');
    
    // XAI and summary elements
    const shapContainer = document.getElementById('shap-container');
    const clinicalSummaryText = document.getElementById('clinical-summary-text');
    const recommendedActions = document.getElementById('recommended-actions');
    const reportCard = document.querySelector('.report-card');

    let currentReportData = null;

    async function runAnalysis(auto = false) {
        if (!auto) {
            analyzeBtn.textContent = 'Analyzing...';
            analyzeBtn.disabled = true;
        }

        // Collect all data
        const formData = new FormData(form);
        const data = {};
        formData.forEach((value, key) => {
            if (key !== 'patient_id' && key !== 'doctor_name' && key !== 'date') {
                data[key] = parseFloat(value);
            }
        });

        const metadata = {
            patient_id: formData.get('patient_id'),
            doctor_name: formData.get('doctor_name'),
            date: formData.get('date')
        };

        try {
            const response = await fetch('/api/analyze_ctg', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ features: data, metadata: metadata })
            });
            
            if (!response.ok) throw new Error('Analysis failed');
            
            const result = await response.json();
            
            // Save current data globally for action buttons
            currentReportData = {
                metadata: metadata,
                features: data,
                result: result
            };

            // Populate results
            renderResults(result);
            
            // Show result section
            const wasHidden = resultSection.classList.contains('hidden');
            resultSection.classList.remove('hidden');
            
            if (!auto && wasHidden) {
                resultSection.scrollIntoView({ behavior: 'smooth' });
            }

        } catch (error) {
            console.error('Error:', error);
            if (!auto) alert('Failed to analyze CTG data.');
        } finally {
            if (!auto) {
                analyzeBtn.textContent = 'Analyze CTG Data';
                analyzeBtn.disabled = false;
            }
        }
    }

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        await runAnalysis(false);
    });

    let typingTimer;
    const numInputs = form.querySelectorAll('input[type="number"]');
    numInputs.forEach(input => {
        input.addEventListener('input', () => {
            clearTimeout(typingTimer);
            typingTimer = setTimeout(() => {
                let allFilled = true;
                numInputs.forEach(i => {
                    if (i.value === '' || isNaN(parseFloat(i.value))) allFilled = false;
                });
                if (allFilled) {
                    runAnalysis(true);
                }
            }, 500); // Live update half a second after typing stops
        });
    });

    function renderResults(result) {
        // Update basic metrics
        fetalHealthStatus.textContent = result.prediction;
        confidenceScore.textContent = `${result.confidence.toFixed(1)}%`;
        
        // Styling status
        statusIndicator.className = 'status-indicator';
        if (result.raw_prediction_idx === 0 || result.prediction === "Normal") {
            statusIndicator.classList.add('status--normal');
            reportCard.style.borderLeftColor = 'var(--normal-color)';
        } else if (result.raw_prediction_idx === 1 || result.prediction === "Suspect") {
            statusIndicator.classList.add('status--suspect');
            reportCard.style.borderLeftColor = 'var(--suspect-color)';
        } else {
            statusIndicator.classList.add('status--pathological');
            reportCard.style.borderLeftColor = 'var(--pathological-color)';
        }

        // Risk Level styling based on confidence
        riskLevel.textContent = result.risk_level;
        if (result.risk_level.includes('Low')) {
            riskLevel.style.color = 'var(--normal-color)';
        } else if (result.risk_level.includes('Medium')) {
            riskLevel.style.color = 'var(--suspect-color)';
        } else {
            riskLevel.style.color = 'var(--pathological-color)';
        }
            
        clinicalSummaryText.textContent = result.clinical_summary;
        recommendedActions.innerHTML = `<li>${result.suggested_action}</li>`;

        // Render SHAP
        renderShap(result.top_features);
    }

    function renderShap(topFeatures) {
        if (!topFeatures || topFeatures.length === 0) return;
        const maxShap = Math.abs(topFeatures[0].value);

        shapContainer.innerHTML = '';
        topFeatures.forEach(f => {
            const width = Math.max((Math.abs(f.value) / maxShap) * 100, 2); // At least 2%
            
            const row = document.createElement('div');
            row.className = 'shap-row';
            
            // Color based on positive/negative impact
            const barColor = f.value > 0 ? '#ef4444' : '#3b82f6';

            row.innerHTML = `
                <div class="shap-label" title="${f.feature}">${f.feature}</div>
                <div class="shap-bar-container">
                    <div class="shap-bar" style="width: ${width}%; background-color: ${barColor};"></div>
                </div>
                <div class="shap-value">${f.value.toFixed(3)}</div>
            `;
            shapContainer.appendChild(row);
        });
    }

    // Image Upload Handling
    const uploadArea = document.getElementById('upload-area');
    const imageUpload = document.getElementById('ctg-image-upload');
    const uploadPlaceholder = document.getElementById('upload-placeholder');
    const previewContainer = document.getElementById('image-preview-container');
    const graphPreview = document.getElementById('graph-preview');
    const removeBtn = document.getElementById('remove-image-btn');

    // Make div clickable for file selection
    uploadPlaceholder.addEventListener('click', (e) => {
        if(e.target.tagName !== 'BUTTON') {
             imageUpload.click();
        }
    });

    imageUpload.addEventListener('change', function() {
        const file = this.files[0];
        if (file) {
            const reader = new FileReader();
            
            reader.addEventListener('load', function() {
                graphPreview.setAttribute('src', this.result);
                uploadPlaceholder.classList.add('hidden');
                previewContainer.classList.remove('hidden');
                uploadArea.style.padding = '1rem';
            });
            
            reader.readAsDataURL(file);
        }
    });

    removeBtn.addEventListener('click', () => {
        imageUpload.value = '';
        graphPreview.setAttribute('src', '');
        previewContainer.classList.add('hidden');
        uploadPlaceholder.classList.remove('hidden');
        uploadArea.style.padding = '2rem';
    });

    // --- Action Buttons ---
    const downloadPdfBtn = document.getElementById('download-pdf-btn');
    const saveReportBtn = document.getElementById('save-report-btn');
    const actionButtonsContainer = document.getElementById('report-action-buttons');

    downloadPdfBtn.addEventListener('click', () => {
        // Hide UI buttons from being rendered in PDF
        if (actionButtonsContainer) actionButtonsContainer.style.display = 'none';

        const reportElem = document.getElementById('result-section');
        const opt = {
            margin:       0.5,
            filename:     `CTG_Report_${currentReportData?.metadata?.patient_id || 'patient'}.pdf`,
            image:        { type: 'jpeg', quality: 0.98 },
            html2canvas:  { scale: 2, useCORS: true, scrollY: 0 },
            jsPDF:        { unit: 'in', format: 'letter', orientation: 'portrait' }
        };
        
        html2pdf().set(opt).from(reportElem).save().then(() => {
            // Restore visibility after generation
            if (actionButtonsContainer) actionButtonsContainer.style.display = 'flex';
        });
    });

    async function saveReportData() {
        if (!currentReportData) {
            alert("Please analyze CTG data first.");
            return false;
        }
        
        let imageBase64 = null;
        if (graphPreview.src && graphPreview.src.startsWith('data:image')) {
            imageBase64 = graphPreview.src;
        }
        
        const payload = { ...currentReportData, image_data: imageBase64 };
        
        saveReportBtn.textContent = 'Saving...';
        try {
            const response = await fetch('/api/save_report', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            if (response.ok) {
                alert("Patient Record Saved Successfully!");
                saveReportBtn.textContent = 'Saved!';
                return true;
            } else {
                throw new Error("Server error");
            }
        } catch (error) {
            console.error(error);
            alert("Error saving record.");
            saveReportBtn.textContent = 'Save Patient Record';
            return false;
        }
    }

    saveReportBtn.addEventListener('click', saveReportData);
});
