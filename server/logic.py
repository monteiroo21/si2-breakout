__author__ = "Mário Antunes"
__version__ = "1.1.0"
__email__ = "mario.antunes@ua.pt"
__status__ = "Development"

import math
import random
from typing import Dict, List, Any
import numpy as np

class Brick:
    def __init__(self, index: int, left: float, top: float, width: float, height: float):
        self.index = index
        self.left = left
        self.top = top
        self.width = width
        self.height = height
        self.right = left + width
        self.bottom = top + height
        self.active = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
            "active": self.active
        }

class Breakout:
    def __init__(self, width: int = 600, height: int = 400):
        self.width = width
        self.height = height
        self.high_score = 0
        
        self.paddle_width = 80.0
        self.paddle_height = 10.0
        self.paddle_y = 380.0
        self.paddle_x = 0.0
        
        self.ball_radius = 8.0
        self.ball_speed = 300.0
        self.ball_x = 0.0
        self.ball_y = 0.0
        self.ball_vx = 0.0
        self.ball_vy = 0.0
        
        self.lives = 3
        self.score = 0
        self.checkpoint_score = 0
        self.game_over = False
        
        self.bricks: List[Brick] = []
        self.brick_array = np.empty((0, 6), dtype=np.float32)
        self.bricks_need_respawn = False
        
        self.reset_game()

    def reset_game(self):
        self.paddle_width = 80.0
        self.paddle_height = 10.0
        self.paddle_y = 380.0
        self.paddle_x = (self.width - self.paddle_width) / 2.0
        
        self.ball_radius = 8.0
        self.ball_speed = 300.0  # Pixels per second
        self.reset_ball()
        
        self.lives = 3
        self.score = 0
        self.checkpoint_score = 0
        self.game_over = False
        
        self.bricks: List[Brick] = []
        self._init_bricks()
        self._sync_bricks_to_numpy()
        self.bricks_need_respawn = False

    def reset_ball(self):
        self.ball_x = self.paddle_x + self.paddle_width / 2.0
        self.ball_y = self.paddle_y - self.ball_radius - 2.0
        
        # Upward-slanted launch: vx = 150, vy = -259.8 (total speed = 300)
        angle = random.uniform(-math.pi / 4.0, math.pi / 4.0)
        self.ball_vx = self.ball_speed * math.sin(angle)
        self.ball_vy = -self.ball_speed * math.cos(angle)

    def _init_bricks(self):
        self.bricks = []
        index = 0
        # Row 1 (top): 5 bricks (offset 105.0)
        for i in range(5):
            self.bricks.append(Brick(index, 105.0 + i * 80.0, 60.0, 70.0, 15.0))
            index += 1
        # Row 2 (middle): 6 bricks (offset 65.0)
        for i in range(6):
            self.bricks.append(Brick(index, 65.0 + i * 80.0, 85.0, 70.0, 15.0))
            index += 1
        # Row 3 (bottom): 5 bricks (offset 105.0)
        for i in range(5):
            self.bricks.append(Brick(index, 105.0 + i * 80.0, 110.0, 70.0, 15.0))
            index += 1

    def _sync_bricks_to_numpy(self):
        data = []
        for b in self.bricks:
            data.append([b.left, b.top, b.right, b.bottom, 1.0 if b.active else 0.0, float(b.index)])
        self.brick_array = np.array(data, dtype=np.float32)

    def move_paddle(self, direction: str):
        if self.game_over:
            return
        speed = 25.0
        if direction == "left" or direction == "WEST":
            self.paddle_x = max(0.0, self.paddle_x - speed)
        elif direction == "right" or direction == "EAST":
            self.paddle_x = min(self.width - self.paddle_width, self.paddle_x + speed)

    def _die(self):
        self.lives -= 1
        self.score = self.checkpoint_score
        if self.lives <= 0:
            self.game_over = True
        else:
            self.paddle_x = (self.width - self.paddle_width) / 2.0
            self.reset_ball()

    def update(self, dt: float):
        if self.game_over:
            return

        # 1. Update ball position
        self.ball_x += self.ball_vx * dt
        self.ball_y += self.ball_vy * dt

        # 2. Wall bounds collision (elastic bounce)
        if self.ball_x - self.ball_radius <= 0.0:
            self.ball_x = self.ball_radius
            self.ball_vx = -self.ball_vx
        elif self.ball_x + self.ball_radius >= self.width:
            self.ball_x = self.width - self.ball_radius
            self.ball_vx = -self.ball_vx

        if self.ball_y - self.ball_radius <= 0.0:
            self.ball_y = self.ball_radius
            self.ball_vy = -self.ball_vy

        # 3. Bottom miss (lose life, reset to last checkpoint)
        if self.ball_y + self.ball_radius > self.height:
            self._die()
            return

        # 4. Paddle collision
        # Bounding box overlap check
        p_left = self.paddle_x
        p_right = self.paddle_x + self.paddle_width
        p_top = self.paddle_y
        p_bottom = self.paddle_y + self.paddle_height
        
        if self.ball_vy > 0.0:
            if (self.ball_x + self.ball_radius >= p_left and self.ball_x - self.ball_radius <= p_right and
                self.ball_y + self.ball_radius >= p_top and self.ball_y - self.ball_radius <= p_bottom):
                
                # 4.1 Region-based bounce angle calculation
                relative_x = self.ball_x - self.paddle_x
                if relative_x < self.paddle_width / 3.0:
                    # Left region: bounce left (angle between -45 and -15 degrees)
                    angle = random.uniform(-math.pi / 4.0, -math.pi / 12.0)
                elif relative_x > 2.0 * self.paddle_width / 3.0:
                    # Right region: bounce right (angle between 15 and 45 degrees)
                    angle = random.uniform(math.pi / 12.0, math.pi / 4.0)
                else:
                    # Center region: bounce nearly straight up (angle between -5 and 5 degrees)
                    angle = random.uniform(-math.pi / 36.0, math.pi / 36.0)

                # 4.2 Conserve Newtonian energy using absolute ball speed
                self.ball_vx = self.ball_speed * math.sin(angle)
                self.ball_vy = -self.ball_speed * math.cos(angle)
                
                # Adjust position to prevent sticky collisions
                self.ball_y = p_top - self.ball_radius

        # 5. Brick collisions using vectorized numpy
        if not self.bricks_need_respawn:
            # Mask active bricks
            active_mask = self.brick_array[:, 4] == 1.0
            if np.any(active_mask):
                lefts = self.brick_array[:, 0]
                tops = self.brick_array[:, 1]
                rights = self.brick_array[:, 2]
                bottoms = self.brick_array[:, 3]
                
                # Ball circular bounding box intersection
                overlap = (active_mask & 
                           (self.ball_x + self.ball_radius >= lefts) &
                           (self.ball_x - self.ball_radius <= rights) &
                           (self.ball_y + self.ball_radius >= tops) &
                           (self.ball_y - self.ball_radius <= bottoms))
                
                if np.any(overlap):
                    # Pick the first hit brick index
                    hit_idx = np.where(overlap)[0][0]
                    brick_idx = int(self.brick_array[hit_idx, 5])
                    
                    # Deactivate in numpy array and bricks list
                    self.brick_array[hit_idx, 4] = 0.0
                    self.bricks[brick_idx].active = False
                    
                    # Specular AABB reflection off the hit brick box
                    b_left = lefts[hit_idx]
                    b_right = rights[hit_idx]
                    b_top = tops[hit_idx]
                    b_bottom = bottoms[hit_idx]
                    
                    # Calculate overlap depths
                    overlap_x = min(self.ball_x + self.ball_radius - b_left, b_right - (self.ball_x - self.ball_radius))
                    overlap_y = min(self.ball_y + self.ball_radius - b_top, b_bottom - (self.ball_y - self.ball_radius))
                    
                    if overlap_x < overlap_y:
                        # Left/Right collision
                        self.ball_vx = -self.ball_vx
                    else:
                        # Top/Bottom collision
                        self.ball_vy = -self.ball_vy
                        
                    # Add points
                    self.score += 3
                    if self.score > self.high_score:
                        self.high_score = self.score
                        
                    # Check if all bricks are now cleared
                    if not np.any(self.brick_array[:, 4] == 1.0):
                        self.score += 100
                        if self.score > self.high_score:
                            self.high_score = self.score
                        self.checkpoint_score = self.score
                        self.bricks_need_respawn = True

        # 6. Infinite Respawn Check: reappears only when the ball falls below the lowest height of the brick pile (y > 140)
        if self.bricks_need_respawn:
            if self.ball_y > 140.0:
                for b in self.bricks:
                    b.active = True
                self._sync_bricks_to_numpy()
                self.bricks_need_respawn = False

    def get_state(self) -> Dict[str, Any]:
        valid_actions = []
        if not self.game_over:
            if self.paddle_x > 0.0:
                valid_actions.append({"action": "move", "direction": "WEST"})
            if self.paddle_x < self.width - self.paddle_width:
                valid_actions.append({"action": "move", "direction": "EAST"})
        return {
            "width": self.width,
            "height": self.height,
            "paddle_x": self.paddle_x,
            "paddle_y": self.paddle_y,
            "paddle_width": self.paddle_width,
            "paddle_height": self.paddle_height,
            "ball_x": self.ball_x,
            "ball_y": self.ball_y,
            "ball_radius": self.ball_radius,
            "lives": self.lives,
            "score": self.score,
            "high_score": self.high_score,
            "game_over": self.game_over,
            "bricks": [b.to_dict() for b in self.bricks if b.active],
            "actions": valid_actions,
            "valid_actions": valid_actions
        }
