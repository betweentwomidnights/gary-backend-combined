const express = require('express');
const fs = require('fs').promises; // Use the promise-based version of fs
const { exec } = require('child_process');
const path = require('path');
const cors = require('cors');
const util = require('util');
const execAsync = util.promisify(exec); // Convert exec to a promise-based function

const app = express();

app.use(cors());
app.use(express.json({ limit: '100mb' }));

// Helper function to handle cleanup
async function cleanup(files) {
    await Promise.all(files.map(file => fs.unlink(file).catch(console.error)));
}

app.post('/combine-audio', async (req, res) => {
    try {
        const audioClips = req.body.audioClips;
        let tempFiles = [];

        for (let i = 0; i < audioClips.length; i++) {
            const buffer = Buffer.from(audioClips[i], 'base64');
            const tempFilePath = path.join(__dirname, `tempAudio${i}.wav`);
            await fs.writeFile(tempFilePath, buffer);
            tempFiles.push(tempFilePath);
        }

        const combinedFilePath = path.join(__dirname, 'combinedAudio.mp3');
        const ffmpegCommand = `ffmpeg -y ${tempFiles.map(file => `-i ${file}`).join(' ')} -filter_complex concat=n=${tempFiles.length}:v=0:a=1 -acodec libmp3lame ${combinedFilePath}`;
        await execAsync(ffmpegCommand);

        const combinedAudio = await fs.readFile(combinedFilePath);
        res.send(Buffer.from(combinedAudio).toString('base64'));

        await cleanup([...tempFiles, combinedFilePath]);
    } catch (error) {
        console.error(`Server error: ${error.message}`);
        res.status(500).send('Server error');
    }
});

app.post('/crop-audio', async (req, res) => {
    const { audioData, end } = req.body;
    const buffer = Buffer.from(audioData, 'base64');
    const tempFilePath = path.join(__dirname, 'tempAudio.wav');
    await fs.writeFile(tempFilePath, buffer);

    const croppedFilePath = path.join(__dirname, 'croppedAudio.mp3');
    const ffmpegCommand = `ffmpeg -y -i ${tempFilePath} -t ${end} -acodec libmp3lame ${croppedFilePath}`;

    try {
        await execAsync(ffmpegCommand);

        const croppedAudio = await fs.readFile(croppedFilePath);
        res.send(Buffer.from(croppedAudio).toString('base64'));

        await cleanup([tempFilePath, croppedFilePath]);
    } catch (error) {
        console.error(`Error cropping audio: ${error.message}`);
        res.status(500).send('Error cropping audio');
    }
});

const PORT = 3000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});