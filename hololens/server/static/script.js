document.addEventListener('DOMContentLoaded', function() {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const clientStream = document.getElementById('client-stream');
    const processedImage = document.getElementById('processed-image');
    const startButton = document.getElementById('start-stream');
    const stopButton = document.getElementById('stop-stream');
    const detectionResults = document.getElementById('detection-results');
    const clientStatus = document.getElementById('client-status');
    const modelStatus = document.getElementById('model-status');
    const serverUrlSpans = document.querySelectorAll('#server-url, #server-url2, #server-url3');

    // Show server URL
    const protocol = window.location.protocol;
    const hostname = window.location.hostname;
    const port = window.location.port;
    const serverUrl = `${protocol}//${hostname}${port ? ':' + port : ''}`;

    serverUrlSpans.forEach(span => {
        span.textContent = serverUrl;
    });

    let stream = null;
    let isStreaming = false;
    let streamInterval = null;
    let isExternalClientConnected = false;

    // Function to start webcam stream from browser
    startButton.addEventListener('click', async function() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                } 
            });
            video.srcObject = stream;
            video.style.display = 'block';
            clientStream.style.display = 'none';

            isStreaming = true;
            startButton.disabled = true;
            stopButton.disabled = false;

            // Start sending frames
            streamInterval = setInterval(sendFrame, 100);  // Send 10 frames per second
        } catch (err) {
            console.error('Error accessing camera:', err);
            alert('Could not access camera. Please make sure it's connected and permissions are granted.');
        }
    });

    // Function to stop webcam stream
    stopButton.addEventListener('click', function() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            video.srcObject = null;
            video.style.display = 'none';
            clientStream.style.display = 'block';
            clearInterval(streamInterval);

            isStreaming = false;
            startButton.disabled = false;
            stopButton.disabled = true;

            detectionResults.textContent = 'Stream stopped';
        }
    });

    // Function to send frame from browser to server
    function sendFrame() {
        if (!isStreaming) return;

        const ctx = canvas.getContext('2d');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        // Draw video frame on canvas
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Convert canvas to base64 image
        const imageData = canvas.toDataURL('image/jpeg', 0.8);

        // Send to server
        fetch('/stream', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `image=${encodeURIComponent(imageData)}`
        })
        .then(response => response.json())
        .then(data => {
            if (data.detection) {
                detectionResults.textContent = data.detection;
            }
        })
        .catch(error => {
            console.error('Error sending frame:', error);
        });
    }

    // Function to update input stream image from external client
    function updateClientStream() {
        // Add timestamp to prevent caching
        const timestamp = new Date().getTime();
        clientStream.src = `/original_frame?t=${timestamp}`;
        
        // Handle load error - show placeholder if needed
        clientStream.onerror = function() {
            if (isExternalClientConnected) {
                this.src = ''; // Clear source to prevent continuous error
                this.alt = 'Waiting for client stream...';
            }
        };
    }
    
    // Function to update processed frame
    function updateProcessedFrame() {
        // Add timestamp to prevent caching
        const timestamp = new Date().getTime();
        processedImage.src = `/processed_frame?t=${timestamp}`;
        
        // Handle load error
        processedImage.onerror = function() {
            this.src = '';
            this.alt = 'No processed frames available';
        };
    }

    // Poll for detection results and server status
    function pollServerStatus() {
        fetch('/status')
            .then(response => response.json())
            .then(data => {
                // Update client connection status
                isExternalClientConnected = data.client_connected;
                
                if (data.client_connected) {
                    clientStatus.textContent = 'Client Status: Connected';
                    clientStatus.className = 'client-connected';
                    
                    if (!isStreaming) {
                        // Show client stream if browser camera is not active
                        clientStream.style.display = 'block';
                    }
                } else {
                    clientStatus.textContent = 'Client Status: Disconnected';
                    clientStatus.className = 'client-disconnected';
                }
                
                // Update model status
                if (data.model_loaded) {
                    modelStatus.textContent = 'Model Status: Loaded';
                }
                
                // Update detection results if available
                if (data.last_detection) {
                    detectionResults.textContent = data.last_detection;
                }
            })
            .catch(error => {
                console.error('Error polling server status:', error);
            });
    }

    // Setup regular polling
    setInterval(pollServerStatus, 1000);
    setInterval(updateClientStream, 100);  // Poll for client stream frames 10 times per second
    setInterval(updateProcessedFrame, 100); // Update processed frames 10 times per second
    
    // Initial status check
    pollServerStatus();
});