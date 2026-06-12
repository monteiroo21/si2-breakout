__author__ = "Mário Antunes"
__version__ = "1.1.0"
__email__ = "mario.antunes@ua.pt"
__status__ = "Development"

import asyncio
import random
from typing import Optional, Dict, Any
from agents.base_agent import BaseAgent

class DummyAgent(BaseAgent):
    async def deliberate(self) -> Optional[Dict[str, Any]]:
        if not self.current_state or self.current_state.get("game_over"):
            return None
        
        valid_actions = self.current_state.get("actions") or []
        if not valid_actions:
            return None
        
        # Randomly choose a valid move or do nothing (stay idle)
        choices = valid_actions + [None]
        return random.choice(choices)

if __name__ == "__main__":
    agent = DummyAgent()
    asyncio.run(agent.run())
