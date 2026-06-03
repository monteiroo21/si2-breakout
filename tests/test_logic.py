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

if __name__ == "__main__":
    unittest.main()
