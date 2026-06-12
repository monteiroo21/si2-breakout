__author__ = "Mário Antunes"
__version__ = "1.1.0"
__email__ = "mario.antunes@ua.pt"
__status__ = "Development"

import unittest
import math
from server.logic import Breakout

class TestBreakoutLogic(unittest.TestCase):
    def setUp(self):
        self.game = Breakout(width=600, height=400)

    def test_initial_state(self):
        self.assertEqual(self.game.lives, 3)
        self.assertEqual(self.game.score, 0)
        self.assertEqual(self.game.checkpoint_score, 0)
        self.assertFalse(self.game.game_over)
        self.assertEqual(len(self.game.bricks), 16)
        for b in self.game.bricks:
            self.assertTrue(b.active)

    def test_move_paddle(self):
        self.game.paddle_x = 10.0
        self.game.move_paddle("WEST")
        self.assertEqual(self.game.paddle_x, 0.0)

        self.game.paddle_x = 510.0
        self.game.move_paddle("EAST")
        self.assertEqual(self.game.paddle_x, 520.0)

    def test_wall_bounce(self):
        # Left wall collision
        self.game.ball_x = 5.0
        self.game.ball_radius = 8.0
        self.game.ball_vx = -100.0
        self.game.ball_vy = 0.0
        self.game.update(0.1)
        self.assertEqual(self.game.ball_vx, 100.0)
        self.assertEqual(self.game.ball_x, 8.0)

        # Right wall collision
        self.game.ball_x = 595.0
        self.game.ball_vx = 100.0
        self.game.ball_vy = 0.0
        self.game.update(0.1)
        self.assertEqual(self.game.ball_vx, -100.0)
        self.assertEqual(self.game.ball_x, 592.0)

        # Ceiling collision
        self.game.ball_y = 5.0
        self.game.ball_vx = 0.0
        self.game.ball_vy = -100.0
        self.game.update(0.1)
        self.assertEqual(self.game.ball_vy, 100.0)
        self.assertEqual(self.game.ball_y, 8.0)

    def test_paddle_bounce_and_conservation(self):
        self.game.paddle_x = 100.0
        self.game.paddle_y = 380.0
        self.game.paddle_width = 80.0
        self.game.paddle_height = 10.0
        
        self.game.ball_x = 140.0
        self.game.ball_y = 373.0
        self.game.ball_vx = 0.0
        self.game.ball_vy = 100.0
        self.game.ball_speed = 300.0

        # Updates position: ball_y becomes 373 + 5 = 378 (overlaps paddle top at 380)
        self.game.update(0.05)
        
        # Velocity vy must reverse
        self.assertLess(self.game.ball_vy, 0.0)
        
        # Magnitude velocity must strictly equal ball_speed
        speed = math.sqrt(self.game.ball_vx**2 + self.game.ball_vy**2)
        self.assertAlmostEqual(speed, 300.0, places=2)
        
        # Position is snapped to paddle top
        self.assertEqual(self.game.ball_y, 372.0)

    def test_brick_collision_and_scoring(self):
        # Disable all but the first brick
        for b in self.game.bricks[1:]:
            b.active = False
        self.game._sync_bricks_to_numpy()

        # The first brick is at left=105, top=60, width=70, height=15 (right=175, bottom=75)
        self.game.ball_x = 140.0
        self.game.ball_y = 80.0
        self.game.ball_vx = 0.0
        self.game.ball_vy = -100.0

        # Update shifts ball to y=75, which overlaps with the bottom edge (y=75)
        self.game.update(0.05)

        self.assertFalse(self.game.bricks[0].active)
        # 3 points for brick + 100 points for clearing the board
        self.assertEqual(self.game.score, 103)
        self.assertEqual(self.game.checkpoint_score, 103)
        self.assertTrue(self.game.bricks_need_respawn)

    def test_die_resets_score_to_checkpoint(self):
        self.game.checkpoint_score = 50
        self.game.score = 75
        self.game.lives = 3
        
        # Position ball at bottom of screen
        self.game.ball_y = 395.0
        self.game.ball_vy = 100.0
        self.game.update(0.1)

        self.assertEqual(self.game.lives, 2)
        self.assertEqual(self.game.score, 50)
        self.assertFalse(self.game.game_over)

        # Die again to game over
        self.game.lives = 1
        self.game.ball_y = 395.0
        self.game.ball_vy = 100.0
        self.game.update(0.1)
        self.assertEqual(self.game.lives, 0)
        self.assertTrue(self.game.game_over)

    def test_brick_respawn(self):
        # Disable all but first brick
        for b in self.game.bricks[1:]:
            b.active = False
        self.game._sync_bricks_to_numpy()

        self.game.ball_x = 140.0
        self.game.ball_y = 80.0
        self.game.ball_vx = 0.0
        self.game.ball_vy = -100.0

        self.game.update(0.05)
        self.assertTrue(self.game.bricks_need_respawn)

        # Moving ball at low height does not respawn bricks
        self.game.ball_y = 100.0
        self.game.update(0.01)
        self.assertTrue(self.game.bricks_need_respawn)

        # Moving ball past lowest brick pile boundary (> 140) respawns them
        self.game.ball_y = 145.0
        self.game.update(0.01)
        self.assertFalse(self.game.bricks_need_respawn)
        for b in self.game.bricks:
            self.assertTrue(b.active)

    def test_paddle_bounce_regions(self):
        self.game.paddle_x = 100.0
        self.game.paddle_y = 380.0
        self.game.paddle_width = 90.0  # clean division by 3: Left 0-30, Center 30-60, Right 60-90
        self.game.paddle_height = 10.0
        self.game.ball_speed = 300.0

        # Left region: hit at relative x = 15.0 (ball_x = 115.0)
        self.game.ball_x = 115.0
        self.game.ball_y = 373.0
        self.game.ball_vx = 0.0
        self.game.ball_vy = 100.0
        self.game.update(0.05)
        # vx must be negative (bounces left)
        self.assertLess(self.game.ball_vx, 0.0)
        self.assertLess(self.game.ball_vy, 0.0)

        # Right region: hit at relative x = 75.0 (ball_x = 175.0)
        self.game.paddle_x = 100.0
        self.game.ball_x = 175.0
        self.game.ball_y = 373.0
        self.game.ball_vx = 0.0
        self.game.ball_vy = 100.0
        self.game.update(0.05)
        # vx must be positive (bounces right)
        self.assertGreater(self.game.ball_vx, 0.0)
        self.assertLess(self.game.ball_vy, 0.0)

        # Center region: hit at relative x = 45.0 (ball_x = 145.0)
        self.game.paddle_x = 100.0
        self.game.ball_x = 145.0
        self.game.ball_y = 373.0
        self.game.ball_vx = 0.0
        self.game.ball_vy = 100.0
        self.game.update(0.05)
        # vx must be small (bounce nearly straight up, |angle| <= 5 deg -> |vx| <= 300 * sin(5 deg) ~ 26.1)
        self.assertLessEqual(abs(self.game.ball_vx), 30.0)
        self.assertLess(self.game.ball_vy, 0.0)

    def test_state_includes_valid_actions(self):
        state = self.game.get_state()
        self.assertIn("actions", state)
        self.assertIn("valid_actions", state)
        self.assertEqual(len(state["actions"]), 2)
        self.assertEqual(state["actions"][0], {"action": "move", "direction": "WEST"})
        self.assertEqual(state["actions"][1], {"action": "move", "direction": "EAST"})

        # If game_over is True, actions must be empty
        self.game.game_over = True
        state_over = self.game.get_state()
        self.assertEqual(len(state_over["actions"]), 0)
        self.assertEqual(len(state_over["valid_actions"]), 0)


if __name__ == "__main__":
    unittest.main()
