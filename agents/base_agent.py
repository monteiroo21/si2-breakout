__author__ = "Mário Antunes"
__version__ = "1.1.0"
__email__ = "mario.antunes@ua.pt"
__status__ = "Development"

import json
import logging
from typing import Any, Dict, Optional

import websockets

logging.basicConfig(level=logging.INFO, format="%(asctime)s - AGENT - %(levelname)s - %(message)s")

class BaseAgent:
    def __init__(self, server_uri: str = "ws://localhost:8765/ws") -> None:
        self.server_uri = server_uri
        self.current_state: Optional[Dict[str, Any]] = None
        self.player_id: Optional[int] = None

    async def run(self) -> None:
        try:
            async with websockets.connect(self.server_uri) as websocket:
                await websocket.send(json.dumps({"client": "agent"}))
                logging.info(f"Connected to {self.server_uri}")

                async for message in websocket:
                    data = json.loads(message)

                    if data.get("type") == "setup":
                        self.player_id = data.get("player_id")
                        logging.info(f"Assigned player ID: {self.player_id}")
                        continue

                    if data.get("type") in ("state", "update"):
                        self.current_state = data
                        action = await self.deliberate()
                        if action:
                            await websocket.send(json.dumps(action))

        except Exception as e:
            logging.error(f"Connection error: {e}")

    async def deliberate(self) -> Optional[Dict[str, Any]]:
        raise NotImplementedError("Subclasses must implement deliberate()")
