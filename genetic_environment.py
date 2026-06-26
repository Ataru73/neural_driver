import numpy as np
import pygame
import math
import gymnasium as gym
from gymnasium import spaces
from environment import generate_track_points, compute_boundaries

class GeneticCarRacingEnv:
    metadata = {"render_modes": ["human"], "render_fps": 60}
    
    def __init__(self, pop_size=30, render_mode=None, num_obstacles=0):
        self.pop_size = pop_size
        self.render_mode = render_mode
        self.num_obstacles = num_obstacles
        self.obstacle_radius = 14.0
        self.obstacles = []
        
        # Window / Screen parameters
        self.width = 1200
        self.height = 900
        
        # Track definition (same as DQN)
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
        outer_start = self.outer_boundary
        outer_end = np.roll(self.outer_boundary, -1, axis=0)
        inner_start = self.inner_boundary
        inner_end = np.roll(self.inner_boundary, -1, axis=0)
        self.Q = np.concatenate([outer_start, inner_start], axis=0)
        self.S = np.concatenate([outer_end - outer_start, inner_end - inner_start], axis=0)
        
        # Car Physics Parameters
        self.max_speed = 8.0
        self.max_steer = math.radians(30.0)
        self.drag = 0.05
        self.car_radius = 12.0
        
        # Raycast parameters
        self.ray_angles = np.array([-90, -70, -45, -30, -15, 0, 15, 30, 45, 70, 90])
        self.num_rays = len(self.ray_angles)
        self.max_ray_len = 250.0
        
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
        
        # State arrays for population
        self.car_pos = np.zeros((self.pop_size, 2), dtype=np.float32)
        self.car_angle = np.zeros(self.pop_size, dtype=np.float32)
        self.car_speed = np.zeros(self.pop_size, dtype=np.float32)
        self.active = np.zeros(self.pop_size, dtype=bool)
        
        self.current_checkpoint = np.zeros(self.pop_size, dtype=int)
        self.next_checkpoint = np.zeros(self.pop_size, dtype=int)
        self.steps_since_progress = np.zeros(self.pop_size, dtype=int)
        self.total_steps = np.zeros(self.pop_size, dtype=int)
        self.lap_count = np.zeros(self.pop_size, dtype=int)
        self.fitness = np.zeros(self.pop_size, dtype=np.float32)
        
        # Rendering
        self.screen = None
        self.clock = None
        self.generation = 1
        
        # Frame stacking for population
        self.history_len = 6
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
        self.lamp_cycle_steps = 60
        self.env_steps = 0
        self.lamp_passed_this_lap = np.zeros(self.pop_size, dtype=bool)
        
    def reset(self, generation=None, random_start=False):
        if generation is not None:
            self.generation = generation
            
        start_idx = 0
        if random_start:
            # Pick one random start checkpoint for the entire population to keep it fair
            start_idx = np.random.randint(0, self.num_checkpoints)
            
        next_idx = (start_idx + 1) % self.num_checkpoints
        tangent = self.centerline[next_idx] - self.centerline[start_idx]
        start_angle = math.atan2(tangent[1], tangent[0])
        
        # Initialize population state arrays
        self.checkpoints_passed = np.zeros(self.pop_size, dtype=int)
        
        for i in range(self.pop_size):
            self.car_pos[i] = np.array(self.centerline[start_idx], dtype=np.float32)
            self.car_angle[i] = start_angle
            self.car_speed[i] = 0.0
            self.active[i] = True
            
            self.current_checkpoint[i] = start_idx
            self.next_checkpoint[i] = next_idx
            self.steps_since_progress[i] = 0
            self.total_steps[i] = 0
            self.lap_count[i] = 0
            self.fitness[i] = 0.0
            
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
                np.random.shuffle(shuffled_cps)
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
                fallback_choices = np.random.choice(valid_cps, size=remaining, replace=True)
                chosen_cps.extend(fallback_choices)
                
            for idx in chosen_cps:
                pt = self.centerline[idx]
                next_pt = self.centerline[(idx + 1) % self.num_checkpoints]
                tangent = next_pt - pt
                length = np.linalg.norm(tangent)
                normal = np.array([-tangent[1], tangent[0]]) / (length if length > 0 else 1.0)
                
                max_offset = (self.track_width / 2.0) - self.obstacle_radius - 12.0
                shift = np.random.uniform(-max_offset, max_offset)
                self.obstacles.append(pt + shift * normal)
                
        # Position street lamp randomly on the track centerline border
        # Select a checkpoint index relative to start_idx to avoid spawning on the car
        lamp_offset = np.random.randint(15, self.num_checkpoints - 15)
        self.lamp_idx = (start_idx + lamp_offset) % self.num_checkpoints
        
        # Calculate normal at lamp_idx to place the lamp on the border
        next_idx = (self.lamp_idx + 1) % self.num_checkpoints
        tangent = self.centerline[next_idx] - self.centerline[self.lamp_idx]
        length = np.linalg.norm(tangent)
        normal = np.array([-tangent[1], tangent[0]]) / (length if length > 0 else 1.0)
        
        # Place lamp on the outer border of the track
        self.lamp_pos = self.centerline[self.lamp_idx] + (self.track_width / 2.0) * normal
        self.lamp_color = 0
        self.env_steps = 0
        self.lamp_passed_this_lap = np.zeros(self.pop_size, dtype=bool)
        
        # Initialize and populate history array for all cars
        self.state_history = np.zeros((self.pop_size, self.history_len, 2 * self.num_rays + 1 + 5), dtype=np.float32)
        for i in range(self.pop_size):
            ray_wall, ray_obs = self._cast_car_rays(i)
            raw_obs = self._build_car_raw_obs(i, ray_wall, ray_obs)
            for h in range(self.history_len):
                self.state_history[i, h] = raw_obs
                
        return self._get_observations()
        
    def step(self, actions):
        """
        Step all cars in the population.
        actions: list or array of action indices of length pop_size
        """
        self.env_steps += 1
        self.lamp_color = (self.env_steps // self.lamp_cycle_steps) % 3
        
        for i in range(self.pop_size):
            if not self.active[i]:
                continue
                
            self.total_steps[i] += 1
            self.steps_since_progress[i] += 1
            
            action = actions[i]
            steer_input, accel_input = self.action_map[action]
            
            # Physics Update
            if accel_input > 0:
                self.car_speed[i] += accel_input * 0.25
            elif accel_input < 0:
                self.car_speed[i] += accel_input * 0.4
                
            self.car_speed[i] -= self.drag * self.car_speed[i]
            self.car_speed[i] = np.clip(self.car_speed[i], 0.0, self.max_speed)
            
            steering_angle = steer_input * self.max_steer
            if self.car_speed[i] > 0.1:
                turn_rate = (self.car_speed[i] / self.max_speed) * steering_angle * 0.15
                self.car_angle[i] += turn_rate
                
            self.car_angle[i] = math.atan2(math.sin(self.car_angle[i]), math.cos(self.car_angle[i]))
            
            self.car_pos[i, 0] += self.car_speed[i] * math.cos(self.car_angle[i])
            self.car_pos[i, 1] += self.car_speed[i] * math.sin(self.car_angle[i])
            
            # Update Progress (Checkpoint tracking)
            self._update_car_progress(i)
            
            # Cast rays to compute collision and distances
            ray_distances_wall, ray_distances_obstacle = self._cast_car_rays(i)
            
            # Update state history for this car
            raw_obs = self._build_car_raw_obs(i, ray_distances_wall, ray_distances_obstacle)
            
            self.state_history[i] = np.roll(self.state_history[i], -1, axis=0)
            self.state_history[i, -1] = raw_obs
            
            # Check collisions
            dist_to_centerline = np.linalg.norm(self.car_pos[i] - self.centerline[self.current_checkpoint[i]])
            off_track = dist_to_centerline > (self.track_width / 2.0 - self.car_radius + 5.0)
            overall_ray_distances = np.minimum(ray_distances_wall, ray_distances_obstacle)
            near_collision = np.any(overall_ray_distances * self.max_ray_len < self.car_radius)
            
            # Check obstacle collisions
            hit_obstacle = False
            for obs_pos in self.obstacles:
                dist_to_obs = np.linalg.norm(self.car_pos[i] - obs_pos)
                if dist_to_obs < (self.car_radius + self.obstacle_radius):
                    hit_obstacle = True
                    break
                    
            # Check collision with red lamp / stop line
            hit_red_lamp = False
            if self.checkpoints_passed[i] == 0:
                self.lamp_passed_this_lap[i] = False
                
            # Compute projection along track direction at the lamp centerline
            prev_idx = (self.lamp_idx - 1) % self.num_checkpoints
            next_idx = (self.lamp_idx + 1) % self.num_checkpoints
            tangent = self.centerline[next_idx] - self.centerline[prev_idx]
            tangent_unit = tangent / np.linalg.norm(tangent)
            dist_long = np.dot(self.car_pos[i] - self.centerline[self.lamp_idx], tangent_unit)
            
            if not self.lamp_passed_this_lap[i]:
                if self.lamp_color == 2: # Red
                    # If front of car (approx -12.0) crosses the stop line
                    if dist_long > -12.0:
                        hit_red_lamp = True
                else: # Green/Yellow
                    # If center of car fully crosses the stop line
                    if dist_long > 5.0:
                        self.lamp_passed_this_lap[i] = True
                        
            collided = off_track or near_collision or hit_obstacle or hit_red_lamp
            
            # Update Fitness
            # Base survival reward
            self.fitness[i] += 0.1
            
            # Check for checkpoint completion
            dist_to_next_cp = np.linalg.norm(self.car_pos[i] - self.centerline[self.next_checkpoint[i]])
            if dist_to_next_cp < (self.track_width * 0.8) and self.current_checkpoint[i] == self.next_checkpoint[i]:
                # Fitness gain for checkpoint
                self.fitness[i] += 150.0
                self.next_checkpoint[i] = (self.next_checkpoint[i] + 1) % self.num_checkpoints
                self.steps_since_progress[i] = 0
                self.checkpoints_passed[i] += 1
                
                if self.checkpoints_passed[i] >= self.num_checkpoints:
                    self.lap_count[i] += 1
                    self.checkpoints_passed[i] = 0
                    # Sorter time (fewer steps) = higher fitness.
                    # Base reward is 50000.0, and we subtract 5.0 per step taken.
                    self.fitness[i] = 50000.0 - (self.total_steps[i] * 5.0)
                    self.active[i] = False
                    continue # Skip other deactivation checks since it completed successfully
                    
            # Check deactivation conditions
            if collided:
                self.active[i] = False
                # Penalize crashing slightly
                self.fitness[i] = max(0.0, self.fitness[i] - 100.0)
            elif self.steps_since_progress[i] > 120:
                # Stagnant car
                self.active[i] = False
                self.fitness[i] = max(0.0, self.fitness[i] - 50.0)

                
        # If human rendering mode is set, display it
        if self.render_mode == "human":
            self._render_frame()
            
        observations = self._get_observations()
        done = not np.any(self.active)
        
        return observations, self.fitness, done, {}
        
    def _update_car_progress(self, i):
        num_pts = self.num_checkpoints
        min_dist = float('inf')
        best_idx = self.current_checkpoint[i]
        
        for offset in range(-5, 15):
            idx = (self.current_checkpoint[i] + offset) % num_pts
            dist = np.linalg.norm(self.car_pos[i] - self.centerline[idx])
            if dist < min_dist:
                min_dist = dist
                best_idx = idx
                
        self.current_checkpoint[i] = best_idx
        
    def _cast_car_rays(self, i):
        ray_distances_wall = []
        ray_distances_obstacle = []
        for angle_deg in self.ray_angles:
            theta = self.car_angle[i] + math.radians(angle_deg)
            r = np.array([math.cos(theta), math.sin(theta)]) * self.max_ray_len
            
            denom = r[0] * self.S[:, 1] - r[1] * self.S[:, 0]
            mask = np.abs(denom) > 1e-6
            
            boundary_t = 1.0
            if np.any(mask):
                q_p_x = self.Q[mask, 0] - self.car_pos[i, 0]
                q_p_y = self.Q[mask, 1] - self.car_pos[i, 1]
                
                t_val = (q_p_x * self.S[mask, 1] - q_p_y * self.S[mask, 0]) / denom[mask]
                u_val = (q_p_x * r[1] - q_p_y * r[0]) / denom[mask]
                
                valid = (t_val >= 0.0) & (t_val <= 1.0) & (u_val >= 0.0) & (u_val <= 1.0)
                valid_t = t_val[valid]
                if len(valid_t) > 0:
                    boundary_t = float(np.min(valid_t))
                    
            # Check obstacle intersections
            obstacle_t = 1.0
            for obs_pos in self.obstacles:
                t_obs = self._intersect_ray_circle(self.car_pos[i], r, obs_pos, self.obstacle_radius)
                if t_obs < obstacle_t:
                    obstacle_t = t_obs
                    
            ray_distances_wall.append(boundary_t)
            ray_distances_obstacle.append(obstacle_t)
                
        return np.array(ray_distances_wall, dtype=np.float32), np.array(ray_distances_obstacle, dtype=np.float32)

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
        
    def _build_car_raw_obs(self, idx, ray_wall, ray_obs):
        obs = np.zeros(2 * self.num_rays + 1 + 5, dtype=np.float32)
        obs[:self.num_rays] = ray_wall
        obs[self.num_rays : 2 * self.num_rays] = ray_obs
        obs[2 * self.num_rays] = self.car_speed[idx] / self.max_speed
        
        # Lamp Color One-hot:
        if self.lamp_color == 0:
            obs[2 * self.num_rays + 1] = 1.0
        elif self.lamp_color == 1:
            obs[2 * self.num_rays + 2] = 1.0
        elif self.lamp_color == 2:
            obs[2 * self.num_rays + 3] = 1.0
            
        # Lamp Relative Position:
        target_ref = self.centerline[self.lamp_idx]
        dist_to_lamp = np.linalg.norm(target_ref - self.car_pos[idx])
        obs[2 * self.num_rays + 4] = min(dist_to_lamp / 400.0, 1.0)
        
        dx = target_ref[0] - self.car_pos[idx, 0]
        dy = target_ref[1] - self.car_pos[idx, 1]
        lamp_angle = math.atan2(dy, dx)
        rel_angle = lamp_angle - self.car_angle[idx]
        rel_angle = (rel_angle + math.pi) % (2 * math.pi) - math.pi
        obs[2 * self.num_rays + 5] = rel_angle / math.pi
        return obs

    def _get_observations(self):
        obs = np.zeros((self.pop_size, self.history_len * (2 * self.num_rays + 1 + 5)), dtype=np.float32)
        for i in range(self.pop_size):
            if not self.active[i]:
                continue
            obs[i] = self.state_history[i].flatten()
        return obs
        
    def _render_frame(self):
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Neuroevolution / Algoritmo Genetico - Auto da Corsa")
            
        if self.clock is None:
            self.clock = pygame.time.Clock()
            
        # Draw background
        self.screen.fill((34, 139, 34))
        
        # Draw track boundaries
        pygame.draw.polygon(self.screen, (60, 60, 60), self.outer_boundary)
        pygame.draw.polygon(self.screen, (34, 139, 34), self.inner_boundary)
        pygame.draw.polygon(self.screen, (220, 220, 220), self.outer_boundary, 3)
        pygame.draw.polygon(self.screen, (220, 220, 220), self.inner_boundary, 3)
        
        # Draw Obstacles
        for obs_pos in self.obstacles:
            pygame.draw.circle(self.screen, (220, 50, 50), obs_pos.astype(int), int(self.obstacle_radius))
            pygame.draw.circle(self.screen, (240, 240, 240), obs_pos.astype(int), int(self.obstacle_radius * 0.6))
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
        
        # Find leading active car (greatest checkpoint or highest fitness)
        leader_idx = -1
        max_fit = -1.0
        for i in range(self.pop_size):
            if self.active[i] and self.fitness[i] > max_fit:
                max_fit = self.fitness[i]
                leader_idx = i
                
        # Draw all cars (faded red for dead, bright yellow for active, blue/cyan for leader)
        for i in range(self.pop_size):
            pos = self.car_pos[i].astype(int)
            angle = self.car_angle[i]
            
            # Skip drawing cars that died long ago at the starting line to avoid clutter
            if not self.active[i] and self.fitness[i] < 50.0:
                continue
                
            car_w, car_l = int(self.car_radius * 1.2), int(self.car_radius * 2.2)
            car_surf = pygame.Surface((car_l, car_w), pygame.SRCALPHA)
            
            # Determine color
            if i == leader_idx:
                color = (0, 191, 255) # Deep Sky Blue for leader
            elif self.active[i]:
                color = (255, 215, 0) # Gold for normal active cars
            else:
                color = (180, 50, 50, 100) # Semi-transparent Red for crashed cars
                
            pygame.draw.rect(car_surf, color, (0, 0, car_l, car_w), border_radius=4)
            # Windshield
            pygame.draw.rect(car_surf, (30, 30, 30), (int(car_l * 0.5), int(car_w * 0.15), int(car_l * 0.25), int(car_w * 0.7)))
            
            rot_angle_deg = -math.degrees(angle)
            rot_car = pygame.transform.rotate(car_surf, rot_angle_deg)
            car_rect = rot_car.get_rect(center=pos)
            self.screen.blit(rot_car, car_rect.topleft)
            
        # Draw Raycasts ONLY for the leader (to keep screen clean)
        if leader_idx != -1:
            ray_distances_wall, ray_distances_obstacle = self._cast_car_rays(leader_idx)
            for angle_deg, dist_wall, dist_obs in zip(self.ray_angles, ray_distances_wall, ray_distances_obstacle):
                theta = self.car_angle[leader_idx] + math.radians(angle_deg)
                
                # Wall intersection
                wall_len = dist_wall * self.max_ray_len
                w_x = self.car_pos[leader_idx, 0] + wall_len * math.cos(theta)
                w_y = self.car_pos[leader_idx, 1] + wall_len * math.sin(theta)
                
                color_wall = (int(255 * (1 - dist_wall)), int(255 * dist_wall), 0)
                pygame.draw.line(self.screen, color_wall, self.car_pos[leader_idx].astype(int), (int(w_x), int(w_y)), 1)
                pygame.draw.circle(self.screen, color_wall, (int(w_x), int(w_y)), 3)
                
                # Obstacle intersection (cyan indicator)
                if dist_obs < 1.0:
                    obs_len = dist_obs * self.max_ray_len
                    o_x = self.car_pos[leader_idx, 0] + obs_len * math.cos(theta)
                    o_y = self.car_pos[leader_idx, 1] + obs_len * math.sin(theta)
                    
                    cyan = (0, 255, 255)
                    pygame.draw.line(self.screen, cyan, self.car_pos[leader_idx].astype(int), (int(o_x), int(o_y)), 1)
                    pygame.draw.circle(self.screen, cyan, (int(o_x), int(o_y)), 4)
                
        # Draw HUD info
        active_count = np.sum(self.active)
        best_fitness = np.max(self.fitness) if len(self.fitness) > 0 else 0.0
        
        font = pygame.font.SysFont("Arial", 20)
        gen_text = font.render(f"Generation: {self.generation}", True, (255, 255, 255))
        active_text = font.render(f"Active Cars: {active_count}/{self.pop_size}", True, (255, 255, 255))
        fit_text = font.render(f"Best Fitness: {best_fitness:.1f}", True, (255, 255, 255))
        
        lamp_colors = ["GREEN", "YELLOW", "RED"]
        lamp_rgb = [(0, 255, 0), (255, 255, 0), (255, 50, 50)]
        lamp_text = font.render(f"Lamp: {lamp_colors[self.lamp_color]}", True, lamp_rgb[self.lamp_color])
        
        hud_bg = pygame.Surface((250, 125))
        hud_bg.fill((0, 0, 0))
        hud_bg.set_alpha(150)
        self.screen.blit(hud_bg, (10, 10))
        
        self.screen.blit(gen_text, (20, 20))
        self.screen.blit(active_text, (20, 45))
        self.screen.blit(fit_text, (20, 70))
        self.screen.blit(lamp_text, (20, 95))
        
        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
        
    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None
