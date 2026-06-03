import asyncio
from typing import Optional, Dict, Any
from agents.base_agent import BaseAgent

class DummyAgent(BaseAgent):
    async def deliberate(self) -> Optional[Dict[str, Any]]:
        if not self.current_state or self.current_state.get("game_over"):
            return None
        
        paddle_x = self.current_state.get("paddle_x", 0.0)
        paddle_width = self.current_state.get("paddle_width", 80.0)
        ball_x = self.current_state.get("ball_x", 0.0)
        
        paddle_center = paddle_x + paddle_width / 2.0
        
        # Move towards the ball's x coordinate
        if ball_x < paddle_center - 10.0:
            return {"action": "move", "direction": "WEST"}
        elif ball_x > paddle_center + 10.0:
            return {"action": "move", "direction": "EAST"}
        
        return None

if __name__ == "__main__":
    agent = DummyAgent()
    asyncio.run(agent.run())
