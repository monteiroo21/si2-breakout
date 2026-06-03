import { GameClient } from '/aigf/framework.js';

const canvas = document.getElementById('game-canvas');
const ctx = canvas.getContext('2d');
const scoreEl = document.getElementById('score');
const highScoreEl = document.getElementById('high-score');
const livesEl = document.getElementById('lives');
const statusEl = document.getElementById('status');
const startBtn = document.getElementById('start-btn');
const resetBtn = document.getElementById('reset-btn');

// Nord Palette
const NORD = {
    nord0: "#2e3440",
    nord1: "#3b4252",
    nord2: "#434c5e",
    nord3: "#4c566a",
    nord4: "#d8dee9",
    nord5: "#e5e9f0",
    nord6: "#eceff4",
    nord7: "#8fbcbb",
    nord8: "#88c0d0",
    nord9: "#81a1c1",
    nord10: "#5e81ac",
    nord11: "#bf616a", // red
    nord12: "#d08770", // orange
    nord13: "#ebcb8b", // yellow
    nord14: "#a3be8c", // green
    nord15: "#b48ead"  // purple
};

const client = new GameClient(8765);

client.onSetup = (data) => {
    canvas.width = data.width || 600;
    canvas.height = data.height || 400;
};

client.onUpdate = (data) => {
    scoreEl.innerText = data.score;
    highScoreEl.innerText = data.high_score;
    livesEl.innerText = data.lives;
    
    const state = data._framework.state;
    statusEl.innerText = state;
    statusEl.className = 'badge ' + (state === 'RUNNING' ? 'badge-running' : 'badge-lobby');
    
    draw(data);
};

startBtn.onclick = () => client.sendCommand('START');
resetBtn.onclick = () => client.sendCommand('RESET');

function draw(state) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 1. Draw Bricks
    if (state.bricks) {
        state.bricks.forEach((b) => {
            if (b.active) {
                // Different rows can have slightly different colors for aesthetics
                let brickColor = NORD.nord14; // default green
                if (b.top === 60.0) {
                    brickColor = NORD.nord13; // top row: yellow
                } else if (b.top === 85.0) {
                    brickColor = NORD.nord12; // middle row: orange
                } else if (b.top === 110.0) {
                    brickColor = NORD.nord11; // bottom row: red
                }
                
                ctx.fillStyle = brickColor;
                ctx.fillRect(b.left, b.top, b.width, b.height);
                
                // Brick border
                ctx.strokeStyle = NORD.nord0;
                ctx.lineWidth = 1.5;
                ctx.strokeRect(b.left, b.top, b.width, b.height);
            }
        });
    }

    // 2. Draw Paddle
    ctx.fillStyle = NORD.nord8;
    ctx.fillRect(state.paddle_x, state.paddle_y, state.paddle_width, state.paddle_height);
    
    // Paddle border
    ctx.strokeStyle = NORD.nord10;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(state.paddle_x, state.paddle_y, state.paddle_width, state.paddle_height);

    // 3. Draw Bouncing Ball
    ctx.beginPath();
    ctx.arc(state.ball_x, state.ball_y, state.ball_radius, 0, 2 * Math.PI);
    ctx.fillStyle = NORD.nord6;
    ctx.fill();
    ctx.strokeStyle = NORD.nord4;
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // 4. Game Over Screen Overlay
    if (state.game_over) {
        ctx.fillStyle = "rgba(46, 52, 64, 0.85)"; // Semi-transparent Nord0
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        ctx.fillStyle = NORD.nord11; // Red
        ctx.font = "bold 32px 'Segoe UI', sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("GAME OVER", canvas.width / 2, canvas.height / 2 - 10);

        ctx.fillStyle = NORD.nord6;
        ctx.font = "16px 'Segoe UI', sans-serif";
        ctx.fillText("Press Start to try again", canvas.width / 2, canvas.height / 2 + 30);
    }
}
