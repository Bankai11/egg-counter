document.addEventListener('DOMContentLoaded', () => {
    // Mode Switching
    const btnSingleMode = document.getElementById('btnSingleMode');
    const btnFolderMode = document.getElementById('btnFolderMode');
    const singleModeView = document.getElementById('singleModeView');
    const folderModeView = document.getElementById('folderModeView');
    
    // Sliders
    const confSlider = document.getElementById('confSlider');
    const confVal = document.getElementById('confVal');
    const iouSlider = document.getElementById('iouSlider');
    const iouVal = document.getElementById('iouVal');
    
    // File inputs & Drag/Drop
    const btnUploadTrigger = document.getElementById('btnUploadTrigger');
    const fileInput = document.getElementById('fileInput');
    const dropzone = document.getElementById('dropzone');
    
    // Single Results UI
    const detectionResult = document.getElementById('detectionResult');
    const originalPreview = document.getElementById('originalPreview');
    const annotatedPreview = document.getElementById('annotatedPreview');
    const imageLoading = document.getElementById('imageLoading');
    const resultCount = document.getElementById('resultCount');
    const boxesTable = document.getElementById('boxesTable').querySelector('tbody');
    
    // Folder scanning UI
    const folderPathInput = document.getElementById('folderPathInput');
    const btnScanFolder = document.getElementById('btnScanFolder');
    const batchProgress = document.getElementById('batchProgress');
    const progressBarFill = document.getElementById('progressBarFill');
    const progressStatus = document.getElementById('progressStatus');
    const progressPct = document.getElementById('progressPct');
    const batchResults = document.getElementById('batchResults');
    const batchTotalImages = document.getElementById('batchTotalImages');
    const batchProcessedImages = document.getElementById('batchProcessedImages');
    const batchTotalTrays = document.getElementById('batchTotalTrays');
    const batchFilesList = document.getElementById('batchFilesList');
    
    // Batch Preview Panel
    const previewPlaceholder = document.getElementById('previewPlaceholder');
    const previewDisplay = document.getElementById('previewDisplay');
    const batchSelectedImage = document.getElementById('batchSelectedImage');
    const previewFilename = document.getElementById('previewFilename');
    const previewCount = document.getElementById('previewCount');
    
    // Status panel
    const statusDot = document.getElementById('statusDot');
    const statusText = document.getElementById('statusText');
    const deviceName = document.getElementById('deviceName');
    const modelWeights = document.getElementById('modelWeights');

    let batchDetections = {};

    // 1. Initial status check
    async function checkStatus() {
        try {
            const res = await fetch('/status');
            if (res.ok) {
                const data = await res.json();
                statusDot.className = 'status-dot online';
                statusText.innerText = 'Connected';
                deviceName.innerText = data.device_name || 'CPU';
                
                // Extract only basename for model weights display
                const weightsFile = data.weights_path.replace(/^.*[\\\/]/, '');
                modelWeights.innerText = weightsFile;
            } else {
                throw new Error();
            }
        } catch (e) {
            statusDot.className = 'status-dot offline';
            statusText.innerText = 'Disconnected';
            deviceName.innerText = 'Unknown';
            modelWeights.innerText = 'yolov8n.pt';
        }
    }

    checkStatus();
    // Poll status every 10 seconds
    setInterval(checkStatus, 10000);

    // 2. Mode selectors
    btnSingleMode.addEventListener('click', () => {
        btnSingleMode.classList.add('active');
        btnFolderMode.classList.remove('active');
        singleModeView.classList.add('active');
        folderModeView.classList.remove('active');
    });

    btnFolderMode.addEventListener('click', () => {
        btnFolderMode.classList.add('active');
        btnSingleMode.classList.remove('active');
        folderModeView.classList.add('active');
        singleModeView.classList.remove('active');
    });

    // 3. Slider displays
    confSlider.addEventListener('input', (e) => {
        confVal.innerText = parseFloat(e.target.value).toFixed(2);
    });

    iouSlider.addEventListener('input', (e) => {
        iouVal.innerText = parseFloat(e.target.value).toFixed(2);
    });

    // 4. Single Image Upload Handlers
    btnUploadTrigger.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleSingleImageUpload(e.target.files[0]);
        }
    });

    // Drag & Drop
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('dragover');
        }, false);
    });

    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleSingleImageUpload(files[0]);
        }
    });

    dropzone.addEventListener('click', () => {
        fileInput.click();
    });

    // 5. Run Single Image Inference
    async function handleSingleImageUpload(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please drop an image file (PNG, JPG, etc.)');
            return;
        }

        // Show result preview state
        dropzone.style.display = 'none';
        detectionResult.style.display = 'flex';
        imageLoading.style.display = 'flex';
        
        // Clear previous table
        boxesTable.innerHTML = '';
        resultCount.innerText = '-';
        
        // Show original image preview locally
        const reader = new FileReader();
        reader.onload = (e) => {
            originalPreview.src = e.target.result;
        };
        reader.readAsDataURL(file);

        // Prep data payload
        const formData = new FormData();
        formData.append('file', file);
        formData.append('conf', confSlider.value);
        formData.append('iou', iouSlider.value);

        try {
            const response = await fetch('/predict-image', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Inference failed');
            }

            const data = await response.json();
            
            // Set annotated output image
            annotatedPreview.src = data.annotated_image;
            resultCount.innerText = data.detected_trays;

            // Fill boxes table
            if (data.boxes && data.boxes.length > 0) {
                data.boxes.forEach((box, idx) => {
                    const row = document.createElement('tr');
                    const [x1, y1, x2, y2, conf] = box;
                    
                    row.innerHTML = `
                        <td>${idx + 1}</td>
                        <td><span class="info-valuehighlight">egg_tray</span></td>
                        <td class="mono">[${x1}, ${y1}, ${x2}, ${y2}]</td>
                        <td class="mono font-bold text-cyan" style="color:var(--primary-neon); font-weight:700;">${(conf * 100).toFixed(1)}%</td>
                    `;
                    boxesTable.appendChild(row);
                });
            } else {
                boxesTable.innerHTML = `<tr><td colspan="4" style="text-align:center; color:var(--text-secondary);">No egg trays detected</td></tr>`;
            }

        } catch (error) {
            console.error(error);
            alert(`Error: ${error.message}`);
            // Reset dropzone
            dropzone.style.display = 'flex';
            detectionResult.style.display = 'none';
        } finally {
            imageLoading.style.display = 'none';
        }
    }

    // 6. Local Folder Scan Handlers
    btnScanFolder.addEventListener('click', async () => {
        const folderPath = folderPathInput.value.trim();
        if (!folderPath) {
            alert('Please enter a server directory path.');
            return;
        }

        // Reset display
        batchResults.style.display = 'none';
        batchProgress.style.display = 'block';
        progressBarFill.style.width = '10%';
        progressStatus.innerText = 'Connecting to server and searching folder...';
        progressPct.innerText = '10%';

        // Simulate folder progression safely while the request completes
        let progress = 10;
        const progressTimer = setInterval(() => {
            if (progress < 85) {
                progress += Math.floor(Math.random() * 5) + 2;
                progressBarFill.style.width = `${progress}%`;
                progressPct.innerText = `${progress}%`;
                
                if (progress > 60) {
                    progressStatus.innerText = 'Detecting trays in image files...';
                } else if (progress > 30) {
                    progressStatus.innerText = 'Reading directories and loading files...';
                }
            }
        }, 300);

        try {
            const response = await fetch('/predict-folder', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    folder_path: folderPath,
                    conf: parseFloat(confSlider.value),
                    iou: parseFloat(iouSlider.value)
                })
            });

            clearInterval(progressTimer);

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Failed to scan folder');
            }

            const data = await response.json();
            
            // Set 100% completion
            progressBarFill.style.width = '100%';
            progressPct.innerText = '100%';
            progressStatus.innerText = 'Scan finished successfully!';
            
            setTimeout(() => {
                batchProgress.style.display = 'none';
                displayFolderResults(data);
            }, 600);

        } catch (error) {
            clearInterval(progressTimer);
            console.error(error);
            alert(`Error: ${error.message}`);
            batchProgress.style.display = 'none';
        }
    });

    function displayFolderResults(data) {
        batchResults.style.display = 'flex';
        batchTotalImages.innerText = data.total_images;
        batchProcessedImages.innerText = data.processed_images;
        batchTotalTrays.innerText = data.total_trays;

        batchFilesList.innerHTML = '';
        batchDetections = data.results;

        // Populate files list
        const filenames = Object.keys(data.results);
        if (filenames.length === 0) {
            batchFilesList.innerHTML = `<li style="text-align:center; color:var(--text-secondary);">No images processed</li>`;
            resetBatchPreview();
            return;
        }

        filenames.forEach(filename => {
            const item = data.results[filename];
            const li = document.createElement('li');
            li.dataset.filename = filename;
            li.innerHTML = `
                <span class="file-name" title="${filename}">${filename}</span>
                <span class="file-badge">Trays: ${item.count}</span>
            `;
            
            li.addEventListener('click', () => {
                // Remove previous selected
                document.querySelectorAll('#batchFilesList li').forEach(el => el.classList.remove('selected'));
                li.classList.add('selected');
                showBatchImagePreview(filename);
            });

            batchFilesList.appendChild(li);
        });

        // Auto select first file preview
        batchFilesList.querySelector('li').click();
    }

    function showBatchImagePreview(filename) {
        const item = batchDetections[filename];
        if (!item) return;

        previewPlaceholder.style.display = 'none';
        previewDisplay.style.display = 'flex';
        
        // Show loading state or text
        previewFilename.innerText = filename;
        previewCount.innerText = `Trays Detected: ${item.count}`;
        
        if (item.preview) {
            batchSelectedImage.src = item.preview;
        } else {
            // Fallback text or local mock if preview base64 is missing
            batchSelectedImage.src = '';
            alert('Detailed base64 preview is only available for the first 10 batch images. All outputs are saved locally on the server under outputs/ directory.');
        }
    }

    function resetBatchPreview() {
        previewPlaceholder.style.display = 'flex';
        previewDisplay.style.display = 'none';
        batchSelectedImage.src = '';
    }
});
