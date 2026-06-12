__author__ = "Mário Antunes"
__version__ = "1.1.0"
__email__ = "mario.antunes@ua.pt"
__status__ = "Development"

import asyncio
import json
import select
import sys

termios = None
tty = None
try:
    import termios
    import tty
    has_termios = True
except ImportError:
    has_termios = False

try:
    import websockets
except ImportError:
    print("Error: The 'websockets' library is required to run the agent.")
    print("Please install it: pip install websockets")
    sys.exit(1)

async def receive_loop(websocket):
    try:
        async for message in websocket:
            data = json.loads(message)
            if data.get("type") == "setup":
                print(f"\n[Handshake Complete] Assigned Breakout Player ID: {data.get('player_id')}")
                print("Controls: A to move LEFT, D to move RIGHT, Q to quit.")
                print("="*60)
            elif data.get("type") in ("state", "update"):
                score = data.get("score", 0)
                lives = data.get("lives", 0)
                high_score = data.get("high_score", 0)
                paddle_x = data.get("paddle_x", 0.0)
                ball_x = data.get("ball_x", 0.0)
                ball_y = data.get("ball_y", 0.0)
                game_over = data.get("game_over", False)
                bricks = data.get("bricks", [])

                # Clean screen redraw
                sys.stdout.write("\033[H\033[2J")
                sys.stdout.flush()

                print("="*18 + " BREAKOUT TERMINAL HUD " + "="*18)
                print(f"HIGH SCORE: {high_score:<10} | SCORE: {score:<10} | LIVES: {lives:<5}")
                print("-" * 59)
                print(f"Paddle Position: {paddle_x:<8.1f} | Ball Position: ({ball_x:.1f}, {ball_y:.1f})")
                print(f"Active Bricks  : {len(bricks):<5} / 16")
                if game_over:
                    print("="*18 + " 💥 GAME OVER! 💥 " + "="*18)
                else:
                    print("="*59)
                print("\n[ACTIVE INPUT] Focus this terminal. Press A/D to move paddle. Press Q to quit.")
    except websockets.exceptions.ConnectionClosed:
        print("\nDisconnected from Breakout Server.")

async def send_loop(websocket):
    fd = sys.stdin.fileno() if has_termios else None
    old_settings = None
    if has_termios and fd is not None:
        assert termios is not None
        assert tty is not None
        old_settings = termios.tcgetattr(fd)
        tty.setraw(fd)

    try:
        while True:
            key = ""
            if has_termios and fd is not None:
                rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
                if rlist:
                    key = sys.stdin.read(1)
            else:
                line = sys.stdin.readline().strip().lower()
                if line == "a":
                    key = "a"
                elif line == "d":
                    key = "d"
                elif line == "q":
                    key = "q"

            if key:
                if key.lower() == "q":
                    break
                if key.lower() == "a":
                    await websocket.send(json.dumps({"action": "move", "direction": "WEST"}))
                elif key.lower() == "d":
                    await websocket.send(json.dumps({"action": "move", "direction": "EAST"}))
            await asyncio.sleep(0.02)
    finally:
        if has_termios and fd is not None and old_settings is not None:
            assert termios is not None
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        print("\nExiting Manual Agent...")

async def main():
    url = "ws://localhost:8765/ws"
    print(f"Connecting to Breakout Server on {url}...")
    try:
        async with websockets.connect(url) as websocket:
            await websocket.send(json.dumps({"client": "agent", "name": "Terminal Breakout"}))
            await asyncio.gather(
                receive_loop(websocket),
                send_loop(websocket)
            )
    except Exception as e:
        print(f"Connection error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBreakout driver exited.")
