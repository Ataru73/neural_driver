import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
import math
from collections import deque

def get_catmull_rom_point(p0, p1, p2, p3, t):
    """
    Calculates a point on a Catmull-Rom spline.
    """
    return 0.5 * (
        (2 * p1) +
        (-p0 + p2) * t +
        (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2 +
        (-p0 + 3 * p1 - 3 * p2 + p3) * t**3
    )

def generate_track_points(control_points, points_per_segment=25):
    """
    Generates a dense set of points along the track from control points.
    """
    track_points = []
    num_cps = len(control_points)
    for i in range(num_cps):
        p0 = control_points[(i - 1) % num_cps]
        p1 = control_points[i]
        p2 = control_points[(i + 1) % num_cps]
        p3 = control_points[(i + 2) % num_cps]
        
        for t in np.linspace(0, 1, points_per_segment, endpoint=False):
            pt = get_catmull_rom_point(p0, p1, p2, p3, t)
            track_points.append(pt)
            
    return np.array(track_points)

def compute_boundaries(centerline, width=90.0):
    """
    Computes inner and outer boundary coordinates for the track.
    """
    num_pts = len(centerline)
    outer = []
    inner = []
    for i in range(num_pts):
        pt = centerline[i]
        next_pt = centerline[(i + 1) % num_pts]
        
        # Tangent vector
        tangent = next_pt - pt
        length = np.linalg.norm(tangent)
        if length > 0:
            tangent /= length
        else:
            tangent = np.array([1.0, 0.0])
            
        # Normal vector (rotate 90 degrees counter-clockwise)
        normal = np.array([-tangent[1], tangent[0]])
        
        outer.append(pt + (width / 2.0) * normal)
        inner.append(pt - (width / 2.0) * normal)
        
    return np.array(outer), np.array(inner)

class CarRacingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(self, render_mode=None, num_obstacles=0):
        super().__init__()
        self.num_obstacles = num_obstacles
        self.obstacle_radius = 14.0
        self.obstacles = []
        
        # Track definition
        # Window size is 1200 x 900
        self.width = 1200
        self.height = 900
        
        control_points = np.array([
            [150, 450],
            [200, 250],
            [450, 150],
            [750, 200],
            [950, 350],
            [1050, 550],
            [900, 750],
            [650, 700],
            [450, 550],
            [250, 650]
        ])

        
        self.track_width = 90.0
        self.centerline = generate_track_points(control_points, points_per_segment=20)
        self.num_checkpoints = len(self.centerline)
        self.outer_boundary, self.inner_boundary = compute_boundaries(self.centerline, self.track_width)
        
        # Precompute segment boundaries for raycasting
        # Combine inner and outer boundaries into segment format: Q (start point) and S (direction vector)
        outer_start = self.outer_boundary
        outer_end = np.roll(self.outer_boundary, -1, axis=0)
        inner_start = self.inner_boundary
        inner_end = np.roll(self.inner_boundary, -1, axis=0)
        
        self.Q = np.concatenate([outer_start, inner_start], axis=0)
        self.S = np.concatenate([outer_end - outer_start, inner_end - inner_start], axis=0)
        
        # Car Physics Parameters
        self.max_speed = 8.0
        self.max_steer = math.radians(30.0) # 30 degrees
        self.drag = 0.05
        self.car_radius = 12.0 # for collision detection
        
        # Raycast parameters
        self.ray_angles = np.array([-90, -70, -45, -30, -15, 0, 15, 30, 45, 70, 90]) # in degrees
        self.num_rays = len(self.ray_angles)
        self.max_ray_len = 250.0
        
        # Define Action Space: Discrete with 9 combinations
        # Actions: [steering, acceleration]
        # steering: -1 (left), 0 (straight), 1 (right)
        # acceleration: -1 (brake/reverse), 0 (coast), 1 (gas)
        self.action_map = {
            0: (-1.0,  1.0), # Turn Left + Gas
            1: ( 0.0,  1.0), # Straight + Gas
            2: ( 1.0,  1.0), # Turn Right + Gas
            3: (-1.0,  0.0), # Turn Left + Coast
            4: ( 0.0,  0.0), # Straight + Coast
            5: ( 1.0,  0.0), # Turn Right + Coast
            6: (-1.0, -0.5), # Turn Left + Brake
            7: ( 0.0, -0.5), # Straight + Brake
            8: ( 1.0, -0.5), # Turn Right + Brake
        }
        self.action_space = spaces.Discrete(9)
        
        # Observation Space:
        # [ray_wall_0..10, ray_obs_0..10, speed] stacked history_len times
        # All values normalized to [0, 1]
        self.history_len = 6
        # Observation Space:
        # [ray_wall_0..10, ray_obs_0..10, speed] + [lamp_green, lamp_yellow, lamp_red, lamp_dist, lamp_angle]
        # stacked history_len times
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.history_len * (2 * self.num_rays + 1 + 5),),
            dtype=np.float32
        )
        
        # Street Lamp attributes
        self.lamp_pos = np.zeros(2)
        self.lamp_idx = 0
        self.lamp_color = 0 # 0: green, 1: yellow, 2: red
        self.lamp_cycle_steps = 60 # change color every 60 steps
        self.lamp_passed_this_lap = False
        
        # Rendering
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        
        # Reset state variables
        self.car_pos = np.zeros(2)
        self.car_angle = 0.0
        self.car_speed = 0.0
        
        self.current_checkpoint = 0
        self.next_checkpoint = 1
        self.steps_since_progress = 0
        self.total_steps = 0
        self.lap_count = 0
        
    def reset(self, seed=None, options=None, random_start=False):
        super().reset(seed=seed)
        
        # Determine starting checkpoint
        start_idx = 0
        if random_start:
            # self.np_random is defined by gymnasium
            start_idx = self.np_random.integers(0, self.num_checkpoints)
            
        self.car_pos = np.array(self.centerline[start_idx], dtype=np.float32)
        
        # Align heading with the direction to the next checkpoint
        next_idx = (start_idx + 1) % self.num_checkpoints
        tangent = self.centerline[next_idx] - self.centerline[start_idx]
        self.car_angle = math.atan2(tangent[1], tangent[0])
        self.car_speed = 0.0
        
        self.current_checkpoint = start_idx
        self.next_checkpoint = next_idx
        self.checkpoints_passed = 0
        self.steps_since_progress = 0
        self.total_steps = 0
        self.lap_count = 0
        
        # Generate obstacles
        self.obstacles = []
        if self.num_obstacles > 0:
            valid_cps = []
            for offset in range(25, self.num_checkpoints - 25):
                idx = (start_idx + offset) % self.num_checkpoints
                valid_cps.append(idx)
            
            chosen_cps = []
            min_dist_cps = max(4, self.num_checkpoints // (2 * self.num_obstacles))
            
            while len(chosen_cps) < self.num_obstacles and min_dist_cps >= 1:
                chosen_cps = []
                shuffled_cps = list(valid_cps)
                self.np_random.shuffle(shuffled_cps)
                for cp in shuffled_cps:
                    if len(chosen_cps) >= self.num_obstacles:
                        break
                    too_close = False
                    for existing in chosen_cps:
                        dist = min(abs(cp - existing), self.num_checkpoints - abs(cp - existing))
                        if dist < min_dist_cps:
                            too_close = True
                            break
                    if not too_close:
                        chosen_cps.append(cp)
                min_dist_cps -= 1
                
            # If still not enough obstacles due to tight constraints, fall back to random choices
            if len(chosen_cps) < self.num_obstacles:
                remaining = self.num_obstacles - len(chosen_cps)
                fallback_choices = self.np_random.choice(valid_cps, size=remaining, replace=True)
                chosen_cps.extend(fallback_choices)
                
            for idx in chosen_cps:
                pt = self.centerline[idx]
                next_pt = self.centerline[(idx + 1) % self.num_checkpoints]
                tangent = next_pt - pt
                length = np.linalg.norm(tangent)
                normal = np.array([-tangent[1], tangent[0]]) / (length if length > 0 else 1.0)
                
                max_offset = (self.track_width / 2.0) - self.obstacle_radius - 12.0
                shift = self.np_random.uniform(-max_offset, max_offset)
                self.obstacles.append(pt + shift * normal)
                
        # Position street lamp randomly on the track centerline border
        # Select a checkpoint index relative to start_idx to avoid spawning on the car
        lamp_offset = self.np_random.integers(15, self.num_checkpoints - 15)
        self.lamp_idx = (start_idx + lamp_offset) % self.num_checkpoints
        
        # Calculate normal at lamp_idx to place the lamp on the border
        next_idx = (self.lamp_idx + 1) % self.num_checkpoints
        tangent = self.centerline[next_idx] - self.centerline[self.lamp_idx]
        length = np.linalg.norm(tangent)
        normal = np.array([-tangent[1], tangent[0]]) / (length if length > 0 else 1.0)
        
        # Place lamp on the outer border of the track
        self.lamp_pos = self.centerline[self.lamp_idx] + (self.track_width / 2.0) * normal
        self.lamp_color = 0
        self.lamp_passed_this_lap = False
                
        raw_obs = self._get_obs()
        self.state_history = deque([raw_obs] * self.history_len, maxlen=self.history_len)
        observation = np.concatenate(self.state_history, dtype=np.float32)
        info = {}
        
        if self.render_mode == "human":
            self._render_frame()
            
        return observation, info
        
    def step(self, action):
        self.total_steps += 1
        self.steps_since_progress += 1
        
        # Decode action
        steer_input, accel_input = self.action_map[action]
        
        # Physics Update (Kinematic Bicycle model approximation)
        # Update speed
        if accel_input > 0:
            self.car_speed += accel_input * 0.25 # Acceleration rate
        elif accel_input < 0:
            self.car_speed += accel_input * 0.4 # Braking rate (stronger deceleration)
            
        # Apply passive drag
        self.car_speed -= self.drag * self.car_speed
        self.car_speed = np.clip(self.car_speed, 0.0, self.max_speed)
        
        # Update angle (steering turn rate is proportional to speed)
        steering_angle = steer_input * self.max_steer
        # Turn rate decreases at higher speed to simulate steering weight, and is 0 when static
        if self.car_speed > 0.1:
            turn_rate = (self.car_speed / self.max_speed) * steering_angle * 0.15
            self.car_angle += turn_rate
            
        # Keep angle normalized between -pi and pi
        self.car_angle = math.atan2(math.sin(self.car_angle), math.cos(self.car_angle))
        
        # Update position
        self.car_pos[0] += self.car_speed * math.cos(self.car_angle)
        self.car_pos[1] += self.car_speed * math.sin(self.car_angle)
        
        # Check closest checkpoint (local search window to avoid skipping/glitching)
        self._update_progress()
        
        # Compute observations and raycasts
        ray_distances_wall, ray_distances_obstacle = self._cast_all_rays()
        
        # Check collision conditions
        # Collision occurs if any ray distance is less than car_radius or if we are too far from centerline
        dist_to_centerline = np.linalg.norm(self.car_pos - self.centerline[self.current_checkpoint])
        off_track = dist_to_centerline > (self.track_width / 2.0 - self.car_radius + 5.0)
        overall_ray_distances = np.minimum(ray_distances_wall, ray_distances_obstacle)
        near_collision = np.any(overall_ray_distances * self.max_ray_len < self.car_radius)
        
        # Check obstacle collisions
        hit_obstacle = False
        for obs_pos in self.obstacles:
            dist_to_obs = np.linalg.norm(self.car_pos - obs_pos)
            if dist_to_obs < (self.car_radius + self.obstacle_radius):
                hit_obstacle = True
                break
                
        # Update lamp color
        self.lamp_color = (self.total_steps // self.lamp_cycle_steps) % 3
        
        # Check collision with red lamp / stop line
        hit_red_lamp = False
        if self.checkpoints_passed == 0:
            self.lamp_passed_this_lap = False
            
        # Compute projection along track direction at the lamp centerline
        prev_idx = (self.lamp_idx - 1) % self.num_checkpoints
        next_idx = (self.lamp_idx + 1) % self.num_checkpoints
        tangent = self.centerline[next_idx] - self.centerline[prev_idx]
        tangent_unit = tangent / np.linalg.norm(tangent)
        dist_long = np.dot(self.car_pos - self.centerline[self.lamp_idx], tangent_unit)
        dist_to_lamp = np.linalg.norm(self.centerline[self.lamp_idx] - self.car_pos)
        
        if not self.lamp_passed_this_lap:
            if dist_to_lamp < 80.0:
                if self.lamp_color == 2: # Red
                    # If front of car (approx -12.0) crosses the stop line
                    if dist_long > -12.0:
                        hit_red_lamp = True
                else: # Green/Yellow
                    # If center of car fully crosses the stop line
                    if dist_long > 5.0:
                        self.lamp_passed_this_lap = True
                    
        collided = off_track or near_collision or hit_obstacle or hit_red_lamp
        
        # Reward function design
        reward = 0.0
        terminated = False
        truncated = False
        
        # 1. Base step penalty to encourage speed and completion
        reward -= 0.1
        
        # Reward shaping for stopping at red lights
        if self.lamp_color == 2 and not self.lamp_passed_this_lap: # Red
            # If the car is approaching the stop line (within 100 pixels, not yet crossed)
            if dist_to_lamp < 100.0 and dist_long < -5.0:
                if self.car_speed < 1.0:
                    reward += 0.4 # Reward for stopping / waiting
                else:
                    reward -= 0.15 # Penalize for driving fast towards red light
        
        # 2. Progress reward (passing checkpoints)
        checkpoint_passed = False
        dist_to_next_cp = np.linalg.norm(self.car_pos - self.centerline[self.next_checkpoint])
        
        # If car gets close to the next checkpoint, reward it and update target checkpoint
        if dist_to_next_cp < (self.track_width * 0.8):
            # Verify direction is forward
            # The current checkpoint should update to the next checkpoint
            if self.current_checkpoint == self.next_checkpoint:
                reward += 15.0
                self.next_checkpoint = (self.next_checkpoint + 1) % self.num_checkpoints
                self.steps_since_progress = 0
                self.checkpoints_passed += 1
                checkpoint_passed = True
                
                # Check for lap completion
                if self.checkpoints_passed >= self.num_checkpoints:
                    self.lap_count += 1
                    self.checkpoints_passed = 0
                    reward += 100.0 # big lap completion reward!
                    # End episode after 2 laps to avoid infinite loops during training
                    if self.lap_count >= 2:
                        terminated = True
        
        # 3. Heading alignment reward: reward aligning with track direction
        # Track heading angle at current checkpoint
        next_idx = (self.current_checkpoint + 1) % self.num_checkpoints
        cp_tangent = self.centerline[next_idx] - self.centerline[self.current_checkpoint]
        track_angle = math.atan2(cp_tangent[1], cp_tangent[0])
        
        angle_diff = math.atan2(math.sin(self.car_angle - track_angle), math.cos(self.car_angle - track_angle))
        alignment = math.cos(angle_diff) # 1.0 if perfectly aligned, -1.0 if facing backwards
        
        # Speed reward (scaled by direction alignment, so going fast backward is penalized)
        reward += (self.car_speed / self.max_speed) * alignment * 0.3
        
        # 4. Collision/Off-track penalty
        if collided:
            reward -= 100.0
            terminated = True
            
        # 5. Stagnation check: terminate if car doesn't make progress for too long
        if self.steps_since_progress > 200:
            reward -= 50.0
            truncated = True
            
        # Prepare observation
        raw_obs = self._build_raw_obs(ray_distances_wall, ray_distances_obstacle)
        
        self.state_history.append(raw_obs)
        observation = np.concatenate(self.state_history, dtype=np.float32)
        
        info = {
            "checkpoint": self.current_checkpoint,
            "next_checkpoint": self.next_checkpoint,
            "laps": self.lap_count,
            "speed": self.car_speed,
            "collided": collided,
            "progress_steps": self.steps_since_progress
        }
        
        if self.render_mode == "human":
            self._render_frame()
            
        return observation, reward, terminated, truncated, info
        
    def _build_raw_obs(self, ray_wall, ray_obs):
        # 2 * self.num_rays + 1 (lidar + speed) + 5 (lamp info) = 28 features
        obs = np.zeros(2 * self.num_rays + 1 + 5, dtype=np.float32)
        obs[:self.num_rays] = ray_wall
        obs[self.num_rays : 2 * self.num_rays] = ray_obs
        obs[2 * self.num_rays] = self.car_speed / self.max_speed
        
        # Calculate relative position to centerline lamp checkpoint
        target_ref = self.centerline[self.lamp_idx]
        dist_to_lamp = np.linalg.norm(target_ref - self.car_pos)
        
        dx = target_ref[0] - self.car_pos[0]
        dy = target_ref[1] - self.car_pos[1]
        lamp_angle = math.atan2(dy, dx)
        rel_angle = lamp_angle - self.car_angle
        rel_angle = (rel_angle + math.pi) % (2 * math.pi) - math.pi
        
        # The car can know the state of the lamp only if it is in front of it and at a distance <= max_ray_len
        if dist_to_lamp <= self.max_ray_len and math.cos(rel_angle) >= 0.0:
            # Lamp Color One-hot: [is_green, is_yellow, is_red]
            if self.lamp_color == 0:
                obs[2 * self.num_rays + 1] = 1.0
            elif self.lamp_color == 1:
                obs[2 * self.num_rays + 2] = 1.0
            elif self.lamp_color == 2:
                obs[2 * self.num_rays + 3] = 1.0
                
            obs[2 * self.num_rays + 4] = dist_to_lamp / self.max_ray_len
            obs[2 * self.num_rays + 5] = rel_angle / math.pi
        else:
            # Default values when lamp is out of range / not in front
            # Color one-hot is [0, 0, 0] (already zeroed by np.zeros)
            obs[2 * self.num_rays + 4] = 1.0 # Max distance
            obs[2 * self.num_rays + 5] = 0.0 # Straight ahead
            
        return obs
        
    def _get_obs(self):
        ray_wall, ray_obs = self._cast_all_rays()
        return self._build_raw_obs(ray_wall, ray_obs)
        
    def _update_progress(self):
        # Local search around current checkpoint
        num_pts = self.num_checkpoints
        min_dist = float('inf')
        best_idx = self.current_checkpoint
        
        # Look 15 checkpoints ahead and 5 checkpoints behind
        for offset in range(-5, 15):
            idx = (self.current_checkpoint + offset) % num_pts
            dist = np.linalg.norm(self.car_pos - self.centerline[idx])
            if dist < min_dist:
                min_dist = dist
                best_idx = idx
                
        self.current_checkpoint = best_idx
        
    def _cast_all_rays(self):
        ray_distances_wall = []
        ray_distances_obstacle = []
        for angle_deg in self.ray_angles:
            # Ray angle in world coordinates
            theta = self.car_angle + math.radians(angle_deg)
            r = np.array([math.cos(theta), math.sin(theta)]) * self.max_ray_len
            
            # Intersection using vectorized logic
            dist_wall, dist_obs = self._cast_ray(self.car_pos, r)
            ray_distances_wall.append(dist_wall)
            ray_distances_obstacle.append(dist_obs)
            
        return np.array(ray_distances_wall, dtype=np.float32), np.array(ray_distances_obstacle, dtype=np.float32)
        
    def _cast_ray(self, p, r):
        # 1. Boundary intersection
        denom = r[0] * self.S[:, 1] - r[1] * self.S[:, 0]
        mask = np.abs(denom) > 1e-6
        
        boundary_t = 1.0
        if np.any(mask):
            q_p_x = self.Q[mask, 0] - p[0]
            q_p_y = self.Q[mask, 1] - p[1]
            
            t_val = (q_p_x * self.S[mask, 1] - q_p_y * self.S[mask, 0]) / denom[mask]
            u_val = (q_p_x * r[1] - q_p_y * r[0]) / denom[mask]
            
            valid = (t_val >= 0.0) & (t_val <= 1.0) & (u_val >= 0.0) & (u_val <= 1.0)
            valid_t = t_val[valid]
            if len(valid_t) > 0:
                boundary_t = float(np.min(valid_t))
                
        # 2. Obstacle intersection
        obstacle_t = 1.0
        for obs_pos in self.obstacles:
            t_obs = self._intersect_ray_circle(p, r, obs_pos, self.obstacle_radius)
            if t_obs < obstacle_t:
                obstacle_t = t_obs
                
        return boundary_t, obstacle_t

    def _intersect_ray_circle(self, p, r, center, radius):
        v = p - center
        a = r[0]**2 + r[1]**2
        b = 2.0 * (v[0]*r[0] + v[1]*r[1])
        c = (v[0]**2 + v[1]**2) - radius**2
        
        disc = b**2 - 4.0 * a * c
        if disc < 0:
            return 1.0
            
        sqrt_disc = math.sqrt(disc)
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)
        
        valid_ts = []
        if 0.0 <= t1 <= 1.0:
            valid_ts.append(t1)
        if 0.0 <= t2 <= 1.0:
            valid_ts.append(t2)
            
        if len(valid_ts) > 0:
            return min(valid_ts)
        return 1.0
        
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
            
    def _render_frame(self):
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.width, self.height))
                pygame.display.set_caption("Neural Network Driving Agent")
            else:
                self.screen = pygame.Surface((self.width, self.height))
                
        if self.clock is None:
            self.clock = pygame.time.Clock()
            
        # Draw background (field)
        self.screen.fill((34, 139, 34)) # Forest Green
        
        # Draw track boundaries
        # Fill track road with dark grey
        pygame.draw.polygon(self.screen, (60, 60, 60), self.outer_boundary)
        pygame.draw.polygon(self.screen, (34, 139, 34), self.inner_boundary) # Grass hole in the middle
        
        # Draw inner and outer borders lines (white / red stripes)
        pygame.draw.polygon(self.screen, (220, 220, 220), self.outer_boundary, 3)
        pygame.draw.polygon(self.screen, (220, 220, 220), self.inner_boundary, 3)
        
        # Draw centerline checkpoints (for debugging/visual reference)
        for i, cp in enumerate(self.centerline):
            # Highlight current and next checkpoints
            if i == self.current_checkpoint:
                pygame.draw.circle(self.screen, (0, 0, 255), cp.astype(int), 4)
            elif i == self.next_checkpoint:
                pygame.draw.circle(self.screen, (255, 165, 0), cp.astype(int), 6)
            else:
                pygame.draw.circle(self.screen, (100, 100, 100), cp.astype(int), 2)
                
        # Draw Obstacles
        for obs_pos in self.obstacles:
            # Draw outer red circle
            pygame.draw.circle(self.screen, (220, 50, 50), obs_pos.astype(int), int(self.obstacle_radius))
            # Draw inner white circle
            pygame.draw.circle(self.screen, (240, 240, 240), obs_pos.astype(int), int(self.obstacle_radius * 0.6))
            # Draw center red dot
            pygame.draw.circle(self.screen, (220, 50, 50), obs_pos.astype(int), int(self.obstacle_radius * 0.2))
            
        # Draw Stop Line
        next_idx = (self.lamp_idx + 1) % self.num_checkpoints
        tangent = self.centerline[next_idx] - self.centerline[self.lamp_idx]
        length = np.linalg.norm(tangent)
        normal = np.array([-tangent[1], tangent[0]]) / (length if length > 0 else 1.0)
        line_start = self.centerline[self.lamp_idx] - (self.track_width / 2.0) * normal
        line_end = self.centerline[self.lamp_idx] + (self.track_width / 2.0) * normal
        pygame.draw.line(self.screen, (240, 240, 240), line_start.astype(int), line_end.astype(int), 4)

        # Draw Street Lamp (Traffic Light)
        lx, ly = int(self.lamp_pos[0]), int(self.lamp_pos[1])
        # Draw pole
        pygame.draw.line(self.screen, (150, 150, 150), (lx, ly), (lx, ly - 20), 4)
        # Draw box
        box_w, box_h = 16, 36
        pygame.draw.rect(self.screen, (30, 30, 30), (lx - box_w//2, ly - 20 - box_h//2, box_w, box_h))
        # Draw lights
        colors = {
            2: ((255, 0, 0), (80, 0, 0)),     # Red
            1: ((255, 255, 0), (80, 80, 0)), # Yellow
            0: ((0, 255, 0), (0, 80, 0))     # Green
        }
        c_red = colors[2][0] if self.lamp_color == 2 else colors[2][1]
        pygame.draw.circle(self.screen, c_red, (lx, ly - 20 - 10), 4)
        c_yel = colors[1][0] if self.lamp_color == 1 else colors[1][1]
        pygame.draw.circle(self.screen, c_yel, (lx, ly - 20), 4)
        c_grn = colors[0][0] if self.lamp_color == 0 else colors[0][1]
        pygame.draw.circle(self.screen, c_grn, (lx, ly - 20 + 10), 4)
            
        # Draw Raycasts
        ray_distances_wall, ray_distances_obstacle = self._cast_all_rays()
        for angle_deg, dist_wall, dist_obs in zip(self.ray_angles, ray_distances_wall, ray_distances_obstacle):
            theta = self.car_angle + math.radians(angle_deg)
            
            # Wall intersection point
            wall_len = dist_wall * self.max_ray_len
            w_x = self.car_pos[0] + wall_len * math.cos(theta)
            w_y = self.car_pos[1] + wall_len * math.sin(theta)
            
            # Draw wall ray
            color_wall = (int(255 * (1 - dist_wall)), int(255 * dist_wall), 0)
            pygame.draw.line(self.screen, color_wall, self.car_pos.astype(int), (int(w_x), int(w_y)), 1)
            pygame.draw.circle(self.screen, color_wall, (int(w_x), int(w_y)), 3)
            
            # Obstacle intersection (cyan indicator)
            if dist_obs < 1.0:
                obs_len = dist_obs * self.max_ray_len
                o_x = self.car_pos[0] + obs_len * math.cos(theta)
                o_y = self.car_pos[1] + obs_len * math.sin(theta)
                
                cyan = (0, 255, 255)
                pygame.draw.line(self.screen, cyan, self.car_pos.astype(int), (int(o_x), int(o_y)), 1)
                pygame.draw.circle(self.screen, cyan, (int(o_x), int(o_y)), 4)
            
        # Draw Car
        # Create a surface for the car to allow rotation
        car_w, car_l = int(self.car_radius * 1.2), int(self.car_radius * 2.2)
        car_surf = pygame.Surface((car_l, car_w), pygame.SRCALPHA)
        
        # Body (Yellow race car)
        pygame.draw.rect(car_surf, (255, 215, 0), (0, 0, car_l, car_w), border_radius=4)
        # Windshield
        pygame.draw.rect(car_surf, (30, 30, 30), (int(car_l * 0.5), int(car_w * 0.15), int(car_l * 0.25), int(car_w * 0.7)))
        # Wheels
        pygame.draw.rect(car_surf, (10, 10, 10), (int(car_l * 0.15), -2, int(car_l * 0.2), 2))
        pygame.draw.rect(car_surf, (10, 10, 10), (int(car_l * 0.15), car_w, int(car_l * 0.2), 2))
        pygame.draw.rect(car_surf, (10, 10, 10), (int(car_l * 0.65), -2, int(car_l * 0.2), 2))
        pygame.draw.rect(car_surf, (10, 10, 10), (int(car_l * 0.65), car_w, int(car_l * 0.2), 2))
        # Spoiler
        pygame.draw.rect(car_surf, (150, 0, 0), (-2, int(car_w * 0.1), 4, int(car_w * 0.8)))
        
        # Rotate car image. Pygame rotates counter-clockwise.
        rot_angle_deg = -math.degrees(self.car_angle)
        rot_car = pygame.transform.rotate(car_surf, rot_angle_deg)
        
        # Position car
        car_rect = rot_car.get_rect(center=self.car_pos.astype(int))
        self.screen.blit(rot_car, car_rect.topleft)
        
        # Draw HUD info
        font = pygame.font.SysFont("Arial", 20)
        speed_text = font.render(f"Speed: {self.car_speed * 10:.1f} km/h", True, (255, 255, 255))
        checkpoint_text = font.render(f"Checkpoint: {self.current_checkpoint}/{self.num_checkpoints}", True, (255, 255, 255))
        laps_text = font.render(f"Laps: {self.lap_count}", True, (255, 255, 255))
        steps_text = font.render(f"Steps: {self.total_steps}", True, (255, 255, 255))
        
        lamp_colors = ["GREEN", "YELLOW", "RED"]
        lamp_rgb = [(0, 255, 0), (255, 255, 0), (255, 50, 50)]
        lamp_text = font.render(f"Lamp: {lamp_colors[self.lamp_color]}", True, lamp_rgb[self.lamp_color])
        
        hud_bg = pygame.Surface((250, 145))
        hud_bg.fill((0, 0, 0))
        hud_bg.set_alpha(150)
        self.screen.blit(hud_bg, (10, 10))
        
        self.screen.blit(speed_text, (20, 20))
        self.screen.blit(checkpoint_text, (20, 45))
        self.screen.blit(laps_text, (20, 70))
        self.screen.blit(steps_text, (20, 95))
        self.screen.blit(lamp_text, (20, 120))
        
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
            
    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None
