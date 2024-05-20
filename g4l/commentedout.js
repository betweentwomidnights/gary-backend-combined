const Max = require('max-api');
const fs = require('fs');
const io = require('socket.io-client');
const socket = io('https://82aa-104-255-9-187.ngrok-free.app');
const path = require('path');

let modelPath = 'facebook/musicgen-small'; // default model path

// Initialize WebSocket connection and setup event listeners
function initSocketConnection() {
    socket.on('connect', () => {
        Max.post('Connected to WebSocket server.');
    });

    socket.on('audio_processed', (data) => {
    Max.post('Audio processing successful.');
    // Decode the base64 string to binary data
    const outputBuffer = Buffer.from(data.audio_data, 'base64');
    // Write the binary data to 'myOutput.wav'
    fs.writeFileSync('C:/gary4live/g4l/myOutput.wav', outputBuffer);
    // Now 'myOutput.wav' should exist
});

    socket.on('status', (data) => {
        Max.post(data.message);
    });

    socket.on('error', (data) => {
        Max.post('Error from WebSocket server: ' + data.message);
        Max.outlet('error', data.message);
    });

    socket.on('disconnect', () => {
        Max.post('Disconnected from WebSocket server.');
    });
}

// Post a message to the Max console when the script is loaded
Max.post(`Loaded the ${path.basename(__filename)} script`);

// Function to handle 'text' message and argument from Max
Max.addHandler('text', (newModelPath) => {
    modelPath = newModelPath;
    Max.post(`Received text, setting modelPath to: ${modelPath}`);
});

// Function to handle 'bang' message from Max
Max.addHandler('bang', () => {
    Max.post(`Received bang, sending audio processing request to WebSocket server.`);
    processAudio();
});

// Function to process audio
function processAudio() {
    // Read the audio file and convert it to base64
    const inputAudioPath = 'C:/gary4live/g4l/myBuffer.wav'; // Replace with the actual path to your audio file
    fs.readFile(inputAudioPath, (err, data) => {
        if (err) {
            Max.post(`Error reading audio file: ${err}`);
            Max.outlet('error', err.toString());
            return;
        }
        const audioData_base64 = data.toString('base64');
        socket.emit('process_audio_request', { audio_data: audioData_base64, model_name: modelPath });
    });
}

// Start the WebSocket connection
initSocketConnection();
