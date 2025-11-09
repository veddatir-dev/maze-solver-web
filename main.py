import pygame
import random
import time
import math
import collections
import heapq
import asyncio # <--- ADDED for web compatibility

# --- 1. CONFIGURATION AND CONSTANTS ---

# Adjusted window size and layout
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 700
MAZE_SIZE = 700  # Fixed size for the maze drawing area (700x700)
ARINA_WIDTH = 160  # Wider width for the extreme-right vertical strip named 'Maze Arina'
UI_PANEL_WIDTH = SCREEN_WIDTH - MAZE_SIZE - ARINA_WIDTH

# Color Themes (same as previous iteration)
DARK_COLORS = {
    'BG': (28, 28, 40), 'WALL': (45, 45, 60), 'PATH': (220, 220, 230),
    'START': (0, 150, 0), 'END': (180, 50, 50), 'PLAYER': (50, 150, 255),
    'TEXT_MAIN': (255, 255, 255), 'TEXT_SUB': (150, 150, 150),
    'BUTTON': (70, 70, 100), 'BUTTON_HOVER': (100, 100, 150),
    'FRONTIER': (255, 165, 0), 'VISITED': (80, 80, 100), 'SOLUTION': (255, 255, 0),
}

LIGHT_COLORS = {
    'BG': (240, 240, 250), 'WALL': (150, 150, 170), 'PATH': (255, 255, 255),
    'START': (50, 180, 50), 'END': (200, 0, 0), 'PLAYER': (0, 100, 200),
    'TEXT_MAIN': (30, 30, 30), 'TEXT_SUB': (100, 100, 100),
    'BUTTON': (200, 200, 220), 'BUTTON_HOVER': (150, 150, 180),
    'FRONTIER': (255, 100, 0), 'VISITED': (180, 180, 200), 'SOLUTION': (255, 165, 0),
}

# Game/Visualization Modes
MODE_GAME = 0
MODE_ALGO = 1
MODE_INFO = 2

# Player Themes (using simple Pygame shapes and colors)
PLAYER_THEMES = collections.OrderedDict([
    ("Dot (Classic)", {'draw_func': 'draw_dot', 'color_override': None}),
    ("Rocket", {'draw_func': 'draw_rocket', 'color_override': (255, 100, 100)}),
    ("Car", {'draw_func': 'draw_car', 'color_override': (100, 100, 255)}),
    ("Pikachu-Style", {'draw_func': 'draw_pikachu', 'color_override': (255, 255, 0)}),
])

# Algorithms and their Descriptions (UPDATED with user-provided detailed content)
ALGORITHMS = collections.OrderedDict([
    ("BFS", {
        "Title": "Breadth-First Search (BFS)",
        "Structure": "Queue (First-In, First-Out or FIFO)",
        "Mechanism": "BFS explores the maze in a wave, guaranteeing the shortest path. It starts by putting the starting cell into the **Queue**. It then dequeues (removes) a cell, marks it as visited, and adds all of its unvisited neighbors to the back of the Queue. This ensures that all cells at the current 'step distance' from the start are explored before moving to the next distance level.",
        "Analogy": "Imagine dropping a pebble in a pool; the search expands outward in ever-increasing, uniform rings."
    }),
    ("DFS", {
        "Title": "Depth-First Search (DFS)",
        "Structure": "Stack (Last-In, First-Out or LIFO)",
        "Mechanism": "DFS explores deep into the maze along a single path. It starts by putting the starting cell onto the **Stack**. It then pops (removes) a cell and immediately puts one of its unvisited neighbors onto the top of the Stack. It continues down this new branch until it hits a dead end, at which point it backtracks (pops cells off) until it finds a cell with an unexplored path.",
        "Analogy": "Following a single hallway as far as you can, and only when you hit a wall, turning around to try the last available door."
    }),
    ("Dijkstra", {
        "Title": "Dijkstra's Algorithm",
        "Structure": "Priority Queue (Min-Heap)",
        "Mechanism": "Dijkstra's finds the path with the **minimum accumulated cost (distance)**. It uses a **Priority Queue** to store all unvisited cells, prioritized by their current shortest distance from the start node (g score). In each step, it extracts the node with the absolute lowest g score, relaxes (updates) the distance of its neighbors if a shorter path through the current node is found, and then adds or updates those neighbors in the Priority Queue.",
        "Analogy": "Always choosing to travel to the nearest point you haven't fully explored yet, ensuring you're building a network of shortest paths outwards."
    }),
    ("A*", {
        "Title": "A* Search",
        "Structure": "Priority Queue (Min-Heap)",
        "Mechanism": "A* is an **improvement over Dijkstra's**. It also uses a **Priority Queue** but prioritizes nodes based on the total estimated cost, $f(n)$, where: $f(n)=g(n)+h(n)$. $g(n)$ is the actual cost from the start (like in Dijkstra's), and $h(n)$ is the heuristic (estimated) cost to the goal (e.g., Manhattan distance). By minimizing $f(n)$, the algorithm is informed and smartly guided toward the goal, making it much faster than Dijkstra's, while still guaranteeing the shortest path (if the heuristic is good).",
        "Analogy": "Like Dijkstra's, but you have a compass pointing toward the exit, allowing you to prioritize paths that move closer to the goal."
    }),
])

# New Level Definitions
LEVELS = [
    (7, "Level 1: Tiny (7x7)"),
    (10, "Level 2: Easy (10x10)"),
    (13, "Level 3: Moderate (13x13)"),
    (17, "Level 4: Challenging (17x17)"),
    (20, "Level 5: Hard (20x20)"),
    (23, "Level 6: Expert (23x23)"),
]

# Player Animation
PLAYER_MOVE_DURATION = 0.2  # seconds

# --- 2. MAZE CLASS AND GENERATION (Unchanged) ---

class Maze:
    """Manages the grid, walls, and maze generation."""
    def __init__(self, cols, rows):
        self.cols = cols
        self.rows = rows
        self.walls = collections.defaultdict(lambda: [True] * 4) # Wall existence: [N, E, S, W]
        self.start = (0, 0)
        self.end = (rows - 1, cols - 1)
        self.generate_maze()

    def generate_maze(self):
        """Generates a perfect maze using the Recursive Backtracker algorithm."""
        grid = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.walls = collections.defaultdict(lambda: [True] * 4)
        stack = [self.start]
        grid[self.start[0]][self.start[1]] = 1
        while stack:
            curr_r, curr_c = stack[-1]
            neighbors = []
            for wall_idx, (dr, dc) in enumerate([(-1, 0), (0, 1), (1, 0), (0, -1)]):
                next_r, next_c = curr_r + dr, curr_c + dc
                if 0 <= next_r < self.rows and 0 <= next_c < self.cols and grid[next_r][next_c] == 0:
                    neighbors.append(((next_r, next_c), wall_idx))
            if neighbors:
                (next_r, next_c), wall_idx = random.choice(neighbors)
                self.walls[(curr_r, curr_c)][wall_idx] = False
                self.walls[(next_r, next_c)][(wall_idx + 2) % 4] = False
                grid[next_r][next_c] = 1
                stack.append((next_r, next_c))
            else:
                stack.pop()

# --- 3. ALGORITHM IMPLEMENTATIONS (Unchanged) ---

def manhattan_distance(p1, p2): return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def get_neighbors(maze, cell):
    r, c = cell
    neighbors = []
    directions = [(-1, 0, 0), (0, 1, 1), (1, 0, 2), (0, -1, 3)]
    for dr, dc, wall_idx in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < maze.rows and 0 <= nc < maze.cols and not maze.walls[(r, c)][wall_idx]:
            neighbors.append((nr, nc))
    return neighbors

def reconstruct_path(parent, start, end):
    path = []
    current = end
    while current and current != start:
        path.append(current)
        current = parent.get(current)
    if current == start: path.append(start)
    path.reverse()
    return path

def solve_bfs(maze):
    queue = collections.deque([maze.start]); visited = {maze.start}; parent = {maze.start: None}; visited_count = 0
    while queue:
        current_cell = queue.popleft(); visited_count += 1
        yield current_cell, list(queue), visited, parent, visited_count
        if current_cell == maze.end: return
        for neighbor in get_neighbors(maze, current_cell):
            if neighbor not in visited:
                visited.add(neighbor); parent[neighbor] = current_cell; queue.append(neighbor)

def solve_dfs(maze):
    stack = [maze.start]; visited = {maze.start}; parent = {maze.start: None}; visited_count = 0
    while stack:
        current_cell = stack.pop(); visited_count += 1
        yield current_cell, stack, visited, parent, visited_count
        if current_cell == maze.end: return
        neighbors = get_neighbors(maze, current_cell); random.shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor); parent[neighbor] = current_cell; stack.append(neighbor)

def solve_dijkstra(maze):
    frontier = [(0, maze.start)]; g_cost = collections.defaultdict(lambda: float('inf')); g_cost[maze.start] = 0
    parent = {maze.start: None}; explored = set(); visited_count = 0
    while frontier:
        current_g, current_cell = heapq.heappop(frontier)
        if current_cell in explored: continue
        explored.add(current_cell); visited_count += 1
        yield current_cell, [c for g, c in frontier], explored, parent, visited_count
        if current_cell == maze.end: return
        for neighbor in get_neighbors(maze, current_cell):
            new_g_cost = current_g + 1
            if new_g_cost < g_cost[neighbor]:
                g_cost[neighbor] = new_g_cost; parent[neighbor] = current_cell
                heapq.heappush(frontier, (new_g_cost, neighbor))

def solve_astar(maze):
    frontier = [(manhattan_distance(maze.start, maze.end), maze.start)]
    g_cost = collections.defaultdict(lambda: float('inf')); g_cost[maze.start] = 0
    parent = {maze.start: None}; explored = set(); visited_count = 0
    while frontier:
        f_cost, current_cell = heapq.heappop(frontier)
        if current_cell in explored: continue
        explored.add(current_cell); visited_count += 1
        yield current_cell, [c for f, c in frontier], explored, parent, visited_count
        if current_cell == maze.end: return
        for neighbor in get_neighbors(maze, current_cell):
            new_g_cost = g_cost[current_cell] + 1
            if new_g_cost < g_cost[neighbor]:
                g_cost[neighbor] = new_g_cost; parent[neighbor] = current_cell
                h_cost = manhattan_distance(neighbor, maze.end)
                f_cost = new_g_cost + h_cost
                heapq.heappush(frontier, (f_cost, neighbor))

# --- 4. PYGAME APPLICATION CLASS ---

class MazeSolverApp:
    def __init__(self):
        pygame.init()
        # Initialize the mixer for sound effects
        # Pygbag note: Sound might be tricky in WASM, but synthesizing simple waves often works.
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        pygame.display.set_caption("Dual-Mode Interactive Maze Solver & Visualizer")
        
        # Pygbag note: Pygame window creation needs to be done early
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.title_font = pygame.font.Font(None, 36)

        # Theme Management
        self.current_theme = 'dark'; self.colors = DARK_COLORS

        # Global State
        self.mode = MODE_GAME; self.running = True; self.current_level_index = 0
        self.sound_enabled = True  # Sound state toggle

        # Game Mode State
        self.player_r, self.player_c = (0, 0); self.game_start_time = time.time()
        self.time_elapsed = 0.0; self.game_won = False
        
        # Player Animation State
        self.player_target_r, self.player_target_c = (0, 0); self.player_start_time = 0.0
        self.player_theme_index = 0
        # Animation timing
        self.animation_start = time.time()
        self.bounce_duration = 0.5  # Duration of one bounce cycle in seconds
        self.float_duration = 1.0   # Duration of floating cycle for rocket
        self.bounce_height = 0.3    # Maximum bounce height relative to cell size
        self.rotation_angle = 0     # Current rotation angle for car wheels
        self.tail_offset = 0        # Pikachu tail wave offset

        # Visualization/Analysis Mode State
        self.selected_algo = "BFS"; self.algo_generator = None
        self.algo_animation_speed = 5; self.algo_frame_counter = 0
        self.algo_metrics = {}; self.current_algo_state = self._reset_algo_state_dict()
        self.speed_levels = {1: "0.2x", 2: "0.5x", 5: "1.0x", 10: "2.0x", 20: "4.0x"}

        # Sound Effects
        self.sfx = self._load_sfx()
        # Arina strip state
        self.arina_enabled = True
        self.arina_buttons = {}
        # Fonts for the Arina strip (distinct look)
        try:
            # Use a slightly fancier but widely available serif for a simple 'fancy' look
            self.arina_title_font = pygame.font.SysFont('georgia', 34, bold=True)
            self.arina_font = pygame.font.SysFont('georgia', 20)
        except Exception:
            # Fallback to existing fonts if SysFont not available
            self.arina_title_font = self.title_font
            self.arina_font = self.font
        # Interactive label rect (set during draw)
        self.arina_label_rect = None
        self.load_level(self.current_level_index)

    def _load_sfx(self):
        """Generates simple synthesized sounds using sine waves."""
        sfx = {}
        sample_rate = 44100
        # Button Click: Short, mid-high tone (C5)
        sfx['click'] = self._create_sine_wave(400, 0.08, sample_rate)
        # Player Move: Very short low tone (C4)
        sfx['move'] = self._create_sine_wave(262, 0.05, sample_rate)
        # Level Win: Ascending arpeggio
        sfx['win'] = self._create_arpeggio([523, 659, 784, 1047], 0.1, sample_rate)
        return sfx

    def _create_sine_wave(self, frequency, duration, sample_rate):
        """Creates a mono sine wave Sound object."""
        n_samples = int(sample_rate * duration)
        sine_wave = [int(32767 * math.sin(2 * math.pi * frequency * i / sample_rate))
                     for i in range(n_samples)]
        # Create a buffer (Pygame expects 16-bit signed integers)
        sound_buffer = bytearray(2 * len(sine_wave))
        for i, val in enumerate(sine_wave):
            # Little-endian bytes for 16-bit
            sound_buffer[2*i] = val & 0xFF
            sound_buffer[2*i + 1] = (val >> 8) & 0xFF
        return pygame.mixer.Sound(sound_buffer)

    def play_sound(self, sound):
        """Play a sound if sound is enabled."""
        if self.sound_enabled:
            sound.play()

    def _create_arpeggio(self, frequencies, note_duration, sample_rate):
        """Creates a Sound object with ascending notes."""
        combined_samples = []
        for freq in frequencies:
            n_samples = int(sample_rate * note_duration)
            sine_wave = [int(32767 * math.sin(2 * math.pi * freq * i / sample_rate))
                         for i in range(n_samples)]
            combined_samples.extend(sine_wave)

        sound_buffer = bytearray(2 * len(combined_samples))
        for i, val in enumerate(combined_samples):
            sound_buffer[2*i] = val & 0xFF
            sound_buffer[2*i + 1] = (val >> 8) & 0xFF
        return pygame.mixer.Sound(sound_buffer)
    
    def _reset_algo_state_dict(self):
        """Returns the initial state dictionary for algorithm visualization."""
        return {
            'current': None, 'frontier': [], 'visited': set(), 'parent': {},
            'path': [], 'time': 0.0, 'steps': 0, 'path_len': 0,
            'is_solving': False, 'solved': False
        }

    # --- Theme Management ---

    def toggle_theme(self):
        """Swaps between dark and light themes."""
        if self.current_theme == 'dark':
            self.current_theme = 'light'
            self.colors = LIGHT_COLORS
        else:
            self.current_theme = 'dark'
            self.colors = DARK_COLORS
            
    # --- Level and Maze Management ---

    def load_level(self, index):
        self.current_level_index = index % len(LEVELS)
        size, _ = LEVELS[self.current_level_index]
        self.maze = Maze(size, size)
        self.cell_size = MAZE_SIZE // size

        # Game Reset
        self.player_r, self.player_c = self.maze.start
        self.player_target_r, self.player_target_c = self.maze.start
        self.player_start_time = 0.0
        self.game_start_time = time.time()
        self.time_elapsed = 0.0
        self.game_won = False

        # Algo Reset
        self.reset_algo_state(); self.algo_metrics = {}

    def reset_algo_state(self):
        self.algo_generator = None
        self.current_algo_state = self._reset_algo_state_dict()

    # --- Drawing Helpers ---

    def draw_cell(self, r, c, color):
        """Draws a single cell in the maze area."""
        x = c * self.cell_size
        y = r * self.cell_size
        pygame.draw.rect(self.screen, color, (x, y, self.cell_size, self.cell_size))

    def draw_maze(self):
        """Draws the core maze structure (walls, start, end)."""
        # Draw cells (Start/End)
        self.draw_cell(self.maze.start[0], self.maze.start[1], self.colors['START'])
        self.draw_cell(self.maze.end[0], self.maze.end[1], self.colors['END'])

        # Draw walls
        for r in range(self.maze.rows):
            for c in range(self.maze.cols):
                x = c * self.cell_size; y = r * self.cell_size
                walls = self.maze.walls[(r, c)]; wall_color = self.colors['WALL']
                
                # N (Top)
                if walls[0]: pygame.draw.line(self.screen, wall_color, (x, y), (x + self.cell_size, y), 2)
                # E (Right)
                if walls[1]: pygame.draw.line(self.screen, wall_color, (x + self.cell_size, y), (x + self.cell_size, y + self.cell_size), 2)
                # S (Bottom) - Only draw outer boundary
                if walls[2] and r == self.maze.rows - 1:
                    pygame.draw.line(self.screen, wall_color, (x, y + self.cell_size), (x + self.cell_size, y + self.cell_size), 2)
                # W (Left) - Only draw outer boundary
                if walls[3] and c == 0:
                    pygame.draw.line(self.screen, wall_color, (x, y), (x, y + self.cell_size), 2)

    def draw_button(self, rect, text, is_active=False):
        """Helper to draw styled buttons."""
        mouse_pos = pygame.mouse.get_pos()
        color = self.colors['BUTTON_HOVER'] if rect.collidepoint(mouse_pos) else self.colors['BUTTON']
        if is_active: color = self.colors['START']
        pygame.draw.rect(self.screen, color, rect, border_radius=5)
        text_surface = self.font.render(text, True, self.colors['TEXT_MAIN'])
        text_rect = text_surface.get_rect(center=rect.center)
        self.screen.blit(text_surface, text_rect)
        return rect
        
    def draw_player(self):
        """Draws the player token, interpolated for smooth movement, using themes."""
        elapsed = time.time() - self.player_start_time
        t = min(1.0, elapsed / PLAYER_MOVE_DURATION)

        r_start, c_start = (self.player_r, self.player_c)
        r_target, c_target = (self.player_target_r, self.player_target_c)

        # Interpolation
        r_current = r_start + (r_target - r_start) * t
        c_current = c_start + (c_target - c_start) * t

        # Pixel Coordinates
        x = c_current * self.cell_size + self.cell_size / 2
        y = r_current * self.cell_size + self.cell_size / 2
        radius = self.cell_size / 3
        
        theme_data = list(PLAYER_THEMES.values())[self.player_theme_index]
        color = theme_data['color_override'] if theme_data['color_override'] else self.colors['PLAYER']
        
        # Call the specific drawing function
        getattr(self, theme_data['draw_func'])(x, y, radius, color)

    # --- Player Icon Drawing Functions (Enhanced) ---

    def draw_dot(self, x, y, radius, color):
        # Calculate bounce offset
        elapsed = time.time() - self.animation_start
        bounce_cycle = (elapsed % self.bounce_duration) / self.bounce_duration
        
        # Sine wave for smooth bouncing
        bounce_offset = math.sin(bounce_cycle * math.pi) * self.bounce_height * self.cell_size
        
        # Apply shadow
        shadow_radius = radius * (1.0 - bounce_offset/(self.cell_size * self.bounce_height * 2))
        shadow_alpha = int(128 * (1.0 - bounce_offset/(self.cell_size * self.bounce_height * 2)))
        shadow_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surface, (*self.colors['WALL'][:3], shadow_alpha), 
                          (0, radius/2, radius * 2, radius))
        self.screen.blit(shadow_surface, 
                        (int(x - radius), int(y + radius/2)),
                        special_flags=pygame.BLEND_ALPHA_SDL2)
        
        # Draw the bouncing ball
        y_pos = y - bounce_offset
        pygame.draw.circle(self.screen, color, (int(x), int(y_pos)), int(radius))
        
        # Add highlight for 3D effect
        highlight_pos = (int(x - radius/3), int(y_pos - radius/3))
        highlight_radius = int(radius/3)
        pygame.draw.circle(self.screen, (255, 255, 255), highlight_pos, highlight_radius)

    def draw_rocket(self, x, y, radius, color):
        elapsed = time.time() - self.animation_start
        float_cycle = (elapsed % self.float_duration) / self.float_duration
        
        # Floating motion
        y_offset = math.sin(float_cycle * 2 * math.pi) * radius * 0.2
        x_offset = math.sin(float_cycle * 4 * math.pi) * radius * 0.1
        
        # Rocket position with offset
        rx = x + x_offset
        ry = y + y_offset
        
        # Draw flame trail
        flame_points = [
            (rx, ry + radius * 1.2),
            (rx - radius * 0.4, ry + radius * 1.6),
            (rx + radius * 0.4, ry + radius * 1.6)
        ]
        flame_color = (255, 165, 0)  # Orange
        flame_intensity = math.sin(elapsed * 10) * 0.5 + 0.5  # Pulsing effect
        flame_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.polygon(flame_surface, (*flame_color, int(255 * flame_intensity)), flame_points)
        self.screen.blit(flame_surface, (rx - radius, ry - radius))
        
        # Main body
        points_body = [(rx, ry - radius), (rx + radius, ry + radius), (rx - radius, ry + radius)]
        pygame.draw.polygon(self.screen, color, points_body)
        
        # Nose cone with shine
        points_nose = [(rx, ry - radius), (rx + radius/3, ry - radius/3), (rx - radius/3, ry - radius/3)]
        pygame.draw.polygon(self.screen, (255, 255, 255), points_nose)
        
        # Animated fins
        fin_color = (50, 150, 255)
        fin_wave = math.sin(elapsed * 8) * radius * 0.1
        fin_points_right = [(rx + radius, ry + radius),
                           (rx + radius + fin_wave, ry + radius/2),
                           (rx + radius/2, ry + radius)]
        fin_points_left = [(rx - radius, ry + radius),
                          (rx - radius - fin_wave, ry + radius/2),
                          (rx - radius/2, ry + radius)]
        pygame.draw.polygon(self.screen, fin_color, fin_points_right)
        pygame.draw.polygon(self.screen, fin_color, fin_points_left)

    def draw_car(self, x, y, radius, color):
        elapsed = time.time() - self.animation_start
        bounce_cycle = (elapsed % (self.bounce_duration * 2)) / (self.bounce_duration * 2)
        
        # Suspension bounce
        bounce_offset = math.sin(bounce_cycle * math.pi) * radius * 0.1
        
        # Update wheel rotation
        self.rotation_angle = (elapsed * 180) % 360  # Rotate wheels
        
        width = radius * 2.5
        height = radius * 1.2
        
        # Draw shadow
        shadow_surface = pygame.Surface((width * 1.2, height), pygame.SRCALPHA)
        shadow_rect = pygame.Rect(width * 0.1, height * 0.8, width, height * 0.2)
        pygame.draw.ellipse(shadow_surface, (*self.colors['WALL'][:3], 100), shadow_rect)
        self.screen.blit(shadow_surface, (x - width/2, y - height/2))
        
        # Main body with bounce
        body_rect = pygame.Rect(x - width/2, y - height/2 - bounce_offset, width, height)
        pygame.draw.rect(self.screen, color, body_rect, border_radius=int(radius/3))
        
        # Window with shine
        window_rect = pygame.Rect(x - width/4, y - height/2 - height/4 - bounce_offset, width/2, height/2)
        pygame.draw.rect(self.screen, (150, 200, 255), window_rect, border_radius=int(radius/4))
        
        # Animated wheels with spokes
        wheel_radius = radius/3
        for wheel_x in [x - width/3, x + width/3]:
            wheel_center = (int(wheel_x), int(y + height/2 - wheel_radius/2 - bounce_offset))
            pygame.draw.circle(self.screen, (0, 0, 0), wheel_center, int(wheel_radius))
            pygame.draw.circle(self.screen, (80, 80, 80), wheel_center, int(wheel_radius * 0.7))
            
            # Draw spokes
            for spoke in range(4):
                angle = math.radians(spoke * 90 + self.rotation_angle)
                spoke_end = (wheel_center[0] + math.cos(angle) * wheel_radius * 0.6,
                           wheel_center[1] + math.sin(angle) * wheel_radius * 0.6)
                pygame.draw.line(self.screen, (200, 200, 200), wheel_center, spoke_end, 2)

    def draw_pikachu(self, x, y, radius, color):
        elapsed = time.time() - self.animation_start
        bounce_cycle = (elapsed % self.bounce_duration) / self.bounce_duration
        
        # Bouncing motion
        bounce_offset = math.sin(bounce_cycle * math.pi) * radius * 0.2
        
        # Ear wave motion
        ear_wave = math.sin(elapsed * 5) * radius * 0.1
        
        # Draw shadow
        shadow_surface = pygame.Surface((radius * 2.5, radius), pygame.SRCALPHA)
        shadow_rect = pygame.Rect(radius * 0.25, radius * 0.8, radius * 2, radius * 0.2)
        pygame.draw.ellipse(shadow_surface, (*self.colors['WALL'][:3], 100), shadow_rect)
        self.screen.blit(shadow_surface, (x - radius * 1.25, y - radius/2))
        
        # Head with bounce
        y_pos = y - bounce_offset
        pygame.draw.circle(self.screen, color, (int(x), int(y_pos)), int(radius))
        
        # Animated ears
        ear_points_left = [(x - radius * 0.5, y_pos - radius * 0.8),
                          (x - radius * 0.7 + ear_wave, y_pos - radius * 1.3),
                          (x - radius * 0.3, y_pos - radius * 0.7)]
        ear_points_right = [(x + radius * 0.5, y_pos - radius * 0.8),
                           (x + radius * 0.7 - ear_wave, y_pos - radius * 1.3),
                           (x + radius * 0.3, y_pos - radius * 0.7)]
        pygame.draw.polygon(self.screen, color, ear_points_left)
        pygame.draw.polygon(self.screen, color, ear_points_right)
        
        # Cheeks with glow effect
        cheek_radius = radius * 0.3
        cheek_glow = math.sin(elapsed * 3) * 30 + 225  # Pulsing red
        cheek_color = (cheek_glow, 0, 0)
        pygame.draw.circle(self.screen, cheek_color, (int(x - radius * 0.6), int(y_pos + radius * 0.3)), int(cheek_radius))
        pygame.draw.circle(self.screen, cheek_color, (int(x + radius * 0.6), int(y_pos + radius * 0.3)), int(cheek_radius))
        
        # Blinking eyes
        eye_radius = radius * 0.15
        blink = (elapsed % 3) > 2.9  # Blink every 3 seconds
        eye_height = eye_radius * (0.1 if blink else 1)
        pygame.draw.ellipse(self.screen, (0, 0, 0),
                          (x - radius * 0.4 - eye_radius, y_pos - radius * 0.3 - eye_height,
                           eye_radius * 2, eye_height * 2))
        pygame.draw.ellipse(self.screen, (0, 0, 0),
                          (x + radius * 0.4 - eye_radius, y_pos - radius * 0.3 - eye_height,
                           eye_radius * 2, eye_height * 2))

    # --- Mode-Specific Draw Functions ---

    def draw_game_mode(self):
        # 1. Draw Maze
        self.draw_maze()
        self.draw_player()

        # 2. Draw HUD (Right Panel)
        hud_area = pygame.Rect(MAZE_SIZE, 0, UI_PANEL_WIDTH, SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, self.colors['BG'], hud_area)
        pygame.draw.line(self.screen, self.colors['WALL'], (MAZE_SIZE, 0), (MAZE_SIZE, SCREEN_HEIGHT), 2)

        x_start = MAZE_SIZE + 20
        y_cursor = 30
        panel_width = 250  # Width for buttons and UI elements

        # Title
        title_surf = self.title_font.render("GAME MODE", True, self.colors['TEXT_MAIN'])
        self.screen.blit(title_surf, (x_start, y_cursor))
        y_cursor += 50

        # Level Selection
        self.screen.blit(self.font.render("Select Level:", True, self.colors['TEXT_SUB']), (x_start, y_cursor))
        y_cursor += 25

        # Create level selection buttons
        level_button_height = 30
        level_spacing = 5
        self.level_buttons = {}
        
        for i, (size, level_name) in enumerate(LEVELS):
            is_current = i == self.current_level_index
            button_rect = pygame.Rect(x_start, y_cursor, panel_width, level_button_height)
            self.level_buttons[i] = button_rect
            self.draw_button(button_rect, level_name, is_active=is_current)
            y_cursor += level_button_height + level_spacing
        
        y_cursor += 20  # Add some spacing after level buttons

        # Timer Info
        time_text = f"{self.time_elapsed:.2f} s"
        self.screen.blit(self.font.render("Time Elapsed:", True, self.colors['TEXT_SUB']), (x_start, y_cursor))
        self.screen.blit(self.font.render(time_text, True, self.colors['SOLUTION']), (x_start, y_cursor + 20))
        y_cursor += 60
        
        # --- Player Theme Control ---
        self.screen.blit(self.font.render("Player Icon:", True, self.colors['TEXT_SUB']), (x_start, y_cursor))
        
        current_theme_name = list(PLAYER_THEMES.keys())[self.player_theme_index]
        self.theme_cycle_rect = pygame.Rect(x_start, y_cursor + 20, 200, 35)
        self.draw_button(self.theme_cycle_rect, current_theme_name)
        y_cursor += 60
        
        # Win Message
        if self.game_won:
            win_msg = f"LEVEL CLEARED in {self.time_elapsed:.2f}s!"
            win_surf = self.title_font.render(win_msg, True, self.colors['START'])
            self.screen.blit(win_surf, (x_start, y_cursor))
            y_cursor += 50
            
            # Next Level Button
            self.next_level_rect = pygame.Rect(x_start, y_cursor, 200, 35)
            self.draw_button(self.next_level_rect, "NEXT LEVEL (Press ENTER)", is_active=True)
            y_cursor += 60

        # Instructions
        self.screen.blit(self.font.render("Controls:", True, self.colors['TEXT_SUB']), (x_start, y_cursor))
        self.screen.blit(self.font.render("Arrow Keys to move.", True, self.colors['TEXT_MAIN']), (x_start, y_cursor + 20))
        y_cursor += 60
        
        # Mode Switch Button
        self.mode_switch_rect = pygame.Rect(x_start, SCREEN_HEIGHT - 100, 200, 35)
        self.draw_button(self.mode_switch_rect, "Switch to Analysis Mode")
        
        # Sound Toggle Button
        sound_text = "Sound: ON" if self.sound_enabled else "Sound: OFF"
        self.sound_toggle_rect = pygame.Rect(x_start, SCREEN_HEIGHT - 150, 200, 35)
        self.draw_button(self.sound_toggle_rect, sound_text)

        # Theme Switch Button
        theme_text = f"Switch to {('Light' if self.current_theme == 'dark' else 'Dark')} Mode"
        self.theme_switch_rect = pygame.Rect(x_start, SCREEN_HEIGHT - 50, 200, 35)
        self.draw_button(self.theme_switch_rect, theme_text)


    def draw_analysis_mode(self):
        # 1. Draw Visualization
        state = self.current_algo_state
        r_current, c_current = state['current'] if state['current'] else (-1, -1)

        for r in range(self.maze.rows):
            for c in range(self.maze.cols):
                cell = (r, c)
                color = self.colors['PATH'] # Default

                if cell in state['path']:
                    color = self.colors['SOLUTION']
                elif cell == (r_current, c_current):
                    color = self.colors['VISITED'] # Currently processing
                elif cell in state['frontier']:
                    color = self.colors['FRONTIER']
                elif cell in state['visited']:
                    color = self.colors['VISITED']

                self.draw_cell(r, c, color)

        # Redraw Start/End over the visualization
        self.draw_maze()

        # 2. Draw HUD (Right Panel)
        hud_area = pygame.Rect(MAZE_SIZE, 0, UI_PANEL_WIDTH, SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, self.colors['BG'], hud_area)
        pygame.draw.line(self.screen, self.colors['WALL'], (MAZE_SIZE, 0), (MAZE_SIZE, SCREEN_HEIGHT), 2)

        x_start = MAZE_SIZE + 20
        y_cursor = 30
        panel_width = 250 # Reduced width for cleaner layout

        # Title
        title_surf = self.title_font.render("ANALYSIS MODE", True, self.colors['TEXT_MAIN'])
        self.screen.blit(title_surf, (x_start, y_cursor))
        y_cursor += 50
        
        # --- Algorithm Selection Buttons ---
        self.algo_buttons = {}
        for i, algo in enumerate(ALGORITHMS.keys()):
            rect = pygame.Rect(x_start + (i % 2) * (panel_width // 2 + 10), y_cursor + (i // 2) * 40, panel_width // 2, 35)
            if not self.current_algo_state['is_solving']:
                self.algo_buttons[algo] = rect
            self.draw_button(rect, algo, is_active=(algo == self.selected_algo))
        
        y_cursor += 90 # Space for 2 rows of buttons

        # --- Run/Reset Button ---
        button_text = "STOP" if self.current_algo_state['is_solving'] else "VISUALIZE / RESTART"
        self.run_reset_rect = pygame.Rect(x_start, y_cursor, panel_width, 35)
        self.draw_button(self.run_reset_rect, button_text, is_active=self.current_algo_state['is_solving'])
        y_cursor += 50

        # --- Speed Control ---
        self.screen.blit(self.font.render("Animation Speed:", True, self.colors['TEXT_SUB']), (x_start, y_cursor))
        y_cursor += 25
        
        # Speed control buttons
        button_width = panel_width // len(self.speed_levels)
        self.speed_buttons = {}
        for i, (speed, label) in enumerate(self.speed_levels.items()):
            rect = pygame.Rect(x_start + i * button_width, y_cursor, button_width - 5, 30)
            is_active = speed == self.algo_animation_speed
            self.speed_buttons[speed] = rect
            self.draw_button(rect, label, is_active=is_active)
        y_cursor += 50
        
        # --- Dynamic Explanation / Algorithm Info Button ---
        self.screen.blit(self.font.render("Selected Algorithm:", True, self.colors['TEXT_SUB']), (x_start, y_cursor))
        y_cursor += 20
        
        # Quick summary for the analysis panel
        mechanism_text = ALGORITHMS[self.selected_algo]['Mechanism']
        quick_explanation = mechanism_text.split('.')[0] + "..." 
        if self.current_algo_state['is_solving']:
            r_current, c_current = self.current_algo_state.get('current', (0,0))
            quick_explanation = f"{self.selected_algo} is exploring cell {r_current, c_current}..."
        
        words = quick_explanation.split(' ')
        lines = []
        current_line = ''
        for word in words:
            if self.font.size(current_line + word)[0] < panel_width:
                current_line += word + ' '
            else:
                lines.append(current_line)
                current_line = word + ' '
        lines.append(current_line)

        for line in lines:
            self.screen.blit(self.font.render(line, True, self.colors['TEXT_MAIN']), (x_start, y_cursor))
            y_cursor += 20
        y_cursor += 20
        
        # Info Button to switch to MODE_INFO
        self.info_button_rect = pygame.Rect(x_start, y_cursor, panel_width, 35)
        self.draw_button(self.info_button_rect, f"More about {self.selected_algo}")
        y_cursor += 50

        # Mode Switch Button (at bottom)
        self.mode_switch_rect = pygame.Rect(x_start, SCREEN_HEIGHT - 100, panel_width, 35)
        self.draw_button(self.mode_switch_rect, "Switch to Game Mode")
        
        # --- Metrics and Comparison ---
        self.screen.blit(self.title_font.render("Comparison Metrics:", True, self.colors['TEXT_MAIN']), (x_start, y_cursor))
        y_cursor += 30

        # Draw header (Fixed positions for alignment)
        header_y = y_cursor
        x_col_algo = x_start
        x_col_steps = x_start + 70
        x_col_len = x_start + 140
        x_col_time = x_start + 220
        
        self.screen.blit(self.font.render("Algo", True, self.colors['TEXT_SUB']), (x_col_algo, header_y))
        self.screen.blit(self.font.render("Visited", True, self.colors['TEXT_SUB']), (x_col_steps, header_y))
        self.screen.blit(self.font.render("Path Len", True, self.colors['TEXT_SUB']), (x_col_len, header_y))
        self.screen.blit(self.font.render("Time(ms)", True, self.colors['TEXT_SUB']), (x_col_time, header_y))
        y_cursor += 25

        # Draw results for each algorithm
        for algo in ALGORITHMS.keys():
            metrics = self.algo_metrics.get(algo, {})
            time_str = f"{metrics.get('time', 0) * 1000:.2f}"
            visited_str = str(metrics.get('steps', 0))
            path_len_str = str(metrics.get('path_len', 0))

            color = self.colors['TEXT_MAIN'] if algo == self.selected_algo else self.colors['TEXT_SUB']

            self.screen.blit(self.font.render(algo, True, color), (x_col_algo, y_cursor))
            self.screen.blit(self.font.render(visited_str, True, color), (x_col_steps, y_cursor))
            self.screen.blit(self.font.render(path_len_str, True, color), (x_col_len, y_cursor))
            self.screen.blit(self.font.render(time_str, True, color), (x_col_time, y_cursor))
            y_cursor += 25
            
        # Theme Switch Button
        theme_text = f"Switch to {('Light' if self.current_theme == 'dark' else 'Dark')} Mode"
        self.theme_switch_rect = pygame.Rect(x_start, SCREEN_HEIGHT - 50, 200, 35)
        self.draw_button(self.theme_switch_rect, theme_text)

    def draw_info_mode(self):
        """Draws the detailed algorithm information screen."""
        self.screen.fill(self.colors['BG'])
        x_start = 50
        y_cursor = 50
        # Ensure info content does not go behind the Arina strip on the right
        max_width = SCREEN_WIDTH - ARINA_WIDTH - 100

        algo_data = ALGORITHMS[self.selected_algo]

        # Title
        title_text = f"Detailed Information: {algo_data['Title']}"
        title_surf = self.title_font.render(title_text, True, self.colors['TEXT_MAIN'])
        self.screen.blit(title_surf, (x_start, y_cursor))
        y_cursor += 60

        # Helper function for drawing wrapped text
        def draw_wrapped_text(text, y, bold_title=None):
            if bold_title:
                self.screen.blit(self.font.render(bold_title, True, self.colors['TEXT_SUB']), (x_start, y))
                y += 20
            
            # Simple word wrapping logic
            words = text.split(' ')
            lines = []
            current_line = ''
            
            for word in words:
                # Check if adding the next word exceeds max_width
                if self.font.size(current_line + word)[0] < max_width - 20: # 20px buffer
                    current_line += word + ' '
                else:
                    lines.append(current_line)
                    current_line = word + ' '
            lines.append(current_line)

            for line in lines:
                # Clip lines so they don't render under the Arina strip
                text_surf = self.font.render(line, True, self.colors['TEXT_MAIN'])
                if text_surf.get_width() > max_width:
                    # Simple clipping by trimming characters (safe, avoids overflow)
                    trimmed = line
                    while self.font.size(trimmed + '...')[0] > max_width and len(trimmed) > 0:
                        trimmed = trimmed[:-1]
                    display_text = trimmed + '...'
                else:
                    display_text = line
                self.screen.blit(self.font.render(display_text, True, self.colors['TEXT_MAIN']), (x_start, y))
                y += 25
            return y + 15

        # 1. Data Structure Used
        y_cursor = draw_wrapped_text(algo_data['Structure'], y_cursor, "Data Structure Used:")

        # 2. How It Works (Mechanism)
        y_cursor = draw_wrapped_text(algo_data['Mechanism'], y_cursor, "How It Works (Mechanism):")

        # 3. Analogy
        y_cursor = draw_wrapped_text(algo_data['Analogy'], y_cursor, "Analogy:")
        

        # Back Button
        self.info_back_rect = pygame.Rect(50, SCREEN_HEIGHT - 70, 200, 40)
        self.draw_button(self.info_back_rect, "Back to Analysis Mode")

    def draw_arina_strip(self):
        """Draws the extreme-right vertical strip labeled 'MAZE ARINA'."""
        x = SCREEN_WIDTH - ARINA_WIDTH
        rect = pygame.Rect(x, 0, ARINA_WIDTH, SCREEN_HEIGHT)
        # Slightly different background to make it distinct (respect theme)
        strip_color = self.colors.get('BUTTON', (90, 90, 120))
        # If disabled, use a muted color
        if not getattr(self, 'arina_enabled', True):
            strip_color = tuple(max(0, c - 40) for c in strip_color)
        pygame.draw.rect(self.screen, strip_color, rect)

        # Draw a top-to-bottom vertical label (each character stacked)
        label = "MAZE ARINA"
        # Prepare simple vertical layout: center horizontally, stack characters from top
        padding_top = 20
        char_spacing = 6
        total_height = 0
        char_surfs = []
        char_sizes = []
        for ch in label:
            surf = self.arina_title_font.render(ch, True, self.colors['TEXT_MAIN'])
            char_surfs.append(surf)
            char_sizes.append((surf.get_width(), surf.get_height()))
            total_height += surf.get_height() + char_spacing

        y = max(padding_top, (SCREEN_HEIGHT - total_height) // 2)
        cx = x + ARINA_WIDTH // 2

        # Build bounding rect for the entire label so we can detect hover/click
        max_char_w = max((w for w, h in char_sizes), default=0)
        label_left = cx - max_char_w // 2 - 6
        label_top = y - 6
        label_w = max_char_w + 12
        label_h = total_height + 12
        self.arina_label_rect = pygame.Rect(label_left, label_top, label_w, label_h)

        mouse_pos = pygame.mouse.get_pos()
        hovered = self.arina_label_rect.collidepoint(mouse_pos)

        # Draw characters with shadow; if hovered, draw glow and change color
        shadow_offset = 3
        for idx, ch in enumerate(label):
            surf = char_surfs[idx]
            # When hovered, make a larger glow by drawing multiple darker layers
            if hovered:
                for i in range(4):
                    alpha = max(20, 120 - i * 25)
                    shadow_surf = surf.copy().convert_alpha()
                    shadow_surf.fill((0, 0, 0, alpha), special_flags=pygame.BLEND_RGBA_MULT)
                    r_sh = shadow_surf.get_rect(center=(cx + shadow_offset + i, y + surf.get_height() // 2 + i))
                    self.screen.blit(shadow_surf, r_sh)
                text_color = self.colors.get('SOLUTION', (255, 255, 0))
            else:
                # Single subtle shadow
                shadow_surf = surf.copy().convert_alpha()
                shadow_surf.fill((0, 0, 0, 100), special_flags=pygame.BLEND_RGBA_MULT)
                r_sh = shadow_surf.get_rect(center=(cx + shadow_offset, y + surf.get_height() // 2 + shadow_offset))
                self.screen.blit(shadow_surf, r_sh)
                text_color = self.colors['TEXT_MAIN']

            # Render and blit the character with the chosen color
            main_surf = self.arina_title_font.render(ch, True, text_color)
            r = main_surf.get_rect(center=(cx, y + main_surf.get_height() // 2))
            self.screen.blit(main_surf, r)
            y += main_surf.get_height() + char_spacing

        # Add small interactive buttons inside the strip (Reset and Toggle)
        btn_w = ARINA_WIDTH - 20
        btn_x = x + 10
        btn_y_start = SCREEN_HEIGHT - 120
        self.arina_buttons = {}

        reset_rect = pygame.Rect(btn_x, btn_y_start, btn_w, 36)
        toggle_rect = pygame.Rect(btn_x, btn_y_start + 46, btn_w, 36)
        self.arina_buttons['reset'] = reset_rect
        self.arina_buttons['toggle'] = toggle_rect

        # Button labels reflect state
        self.draw_button(reset_rect, "RESET MAZE")
        toggle_label = "ARINA: ON" if getattr(self, 'arina_enabled', True) else "ARINA: OFF"
        self.draw_button(toggle_rect, toggle_label, is_active=getattr(self, 'arina_enabled', True))

    # --- Game Mode Logic ---

    def move_player(self, dr, dc):
        if self.game_won or time.time() < self.player_start_time + PLAYER_MOVE_DURATION: return

        r, c = (self.player_target_r, self.player_target_c)
        nr, nc = r + dr, c + dc

        wall_idx = -1
        if dr == -1: wall_idx = 0
        elif dc == 1: wall_idx = 1
        elif dr == 1: wall_idx = 2
        elif dc == -1: wall_idx = 3

        if not (0 <= nr < self.maze.rows and 0 <= nc < self.maze.cols): return
        if wall_idx != -1 and self.maze.walls[(r, c)][wall_idx]: return

        # Play move sound effect
        self.play_sound(self.sfx['move'])

        # Start the animation
        self.player_r, self.player_c = r, c
        self.player_target_r, self.player_target_c = nr, nc
        self.player_start_time = time.time()

        if (nr, nc) == self.maze.end:
            self.check_game_end()

    def check_game_end(self):
        if (self.player_target_r, self.player_target_c) == self.maze.end and not self.game_won:
            self.game_won = True
            self.time_elapsed = time.time() - self.game_start_time
            self.play_sound(self.sfx['win']) # Play win sound
            
    # --- Analysis Mode Logic (Unchanged) ---
    def start_visualization(self):
        self.reset_algo_state()
        self.current_algo_state['is_solving'] = True
        self.algo_start_time = time.time()
        if self.selected_algo == "BFS": self.algo_generator = solve_bfs(self.maze)
        elif self.selected_algo == "DFS": self.algo_generator = solve_dfs(self.maze)
        elif self.selected_algo == "Dijkstra": self.algo_generator = solve_dijkstra(self.maze)
        elif self.selected_algo == "A*": self.algo_generator = solve_astar(self.maze)

    def step_visualization(self):
        if not self.algo_generator: return
        try:
            current, frontier, visited, parent, steps = next(self.algo_generator)
            self.current_algo_state.update({
                'current': current, 'frontier': frontier, 'visited': visited, 'parent': parent, 'steps': steps,
            })
            if current == self.maze.end: self.end_visualization(parent, steps)
        except StopIteration:
            if not self.current_algo_state['solved']: self.end_visualization(self.current_algo_state['parent'], self.current_algo_state['steps'], found=False)

    def end_visualization(self, parent_map, steps, found=True):
        end_time = time.time()
        self.current_algo_state['is_solving'] = False
        self.current_algo_state['solved'] = True
        elapsed_time = end_time - self.algo_start_time
        self.current_algo_state['time'] = elapsed_time
        path = []; path_len = 0
        if found:
            path = reconstruct_path(parent_map, self.maze.start, self.maze.end)
            path_len = len(path) - 1
        self.current_algo_state['path'] = path
        self.current_algo_state['path_len'] = path_len
        self.algo_metrics[self.selected_algo] = {
            'time': elapsed_time, 'steps': steps, 'path_len': path_len,
        }

    # --- Input and Main Loop ---

    def handle_input(self, event):
        if event.type == pygame.QUIT: self.running = False; return

        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = event.pos
            self.play_sound(self.sfx['click'])

            # If click happened inside the Arina strip, handle label and buttons first
            if mouse_pos[0] >= SCREEN_WIDTH - ARINA_WIDTH:
                # Check label click (toggle) if available
                label_rect = getattr(self, 'arina_label_rect', None)
                if label_rect and label_rect.collidepoint(mouse_pos):
                    # Clicking the vertical text toggles the Arina strip
                    self.arina_enabled = not self.arina_enabled
                    self.play_sound(self.sfx['click'])
                    return

                # Otherwise check stored arina buttons
                for name, rect in getattr(self, 'arina_buttons', {}).items():
                    if rect.collidepoint(mouse_pos):
                        if name == 'reset':
                            # Reload current level (regenerate maze)
                            self.load_level(self.current_level_index)
                            return
                        elif name == 'toggle':
                            self.arina_enabled = not self.arina_enabled
                            self.play_sound(self.sfx['click'])
                            return

            if hasattr(self, 'sound_toggle_rect') and self.sound_toggle_rect.collidepoint(mouse_pos):
                self.sound_enabled = not self.sound_enabled
                return
                
            if hasattr(self, 'theme_switch_rect') and self.theme_switch_rect.collidepoint(mouse_pos): self.toggle_theme(); return
            if self.mode == MODE_INFO and hasattr(self, 'info_back_rect') and self.info_back_rect.collidepoint(mouse_pos): self.mode = MODE_ALGO; return
            if hasattr(self, 'mode_switch_rect') and self.mode_switch_rect.collidepoint(mouse_pos):
                if self.mode == MODE_GAME:
                    self.mode = MODE_ALGO
                    self.reset_algo_state()
                elif self.mode == MODE_ALGO:
                    self.mode = MODE_GAME
                    self.game_start_time = time.time()
                elif self.mode == MODE_INFO:
                    self.mode = MODE_GAME
                    self.game_start_time = time.time()
                return

            if self.mode == MODE_GAME:
                if self.game_won and hasattr(self, 'next_level_rect') and self.next_level_rect.collidepoint(mouse_pos):
                    self.load_level(self.current_level_index + 1)
                if hasattr(self, 'theme_cycle_rect') and self.theme_cycle_rect.collidepoint(mouse_pos):
                    self.player_theme_index = (self.player_theme_index + 1) % len(PLAYER_THEMES)
                # Level selection
                if hasattr(self, 'level_buttons'):
                    for level_idx, rect in self.level_buttons.items():
                        if rect.collidepoint(mouse_pos) and not self.game_won:
                            self.load_level(level_idx)

            elif self.mode == MODE_ALGO:
                if hasattr(self, 'info_button_rect') and self.info_button_rect.collidepoint(mouse_pos): self.mode = MODE_INFO; return
                if hasattr(self, 'run_reset_rect') and self.run_reset_rect.collidepoint(mouse_pos):
                    if self.current_algo_state['is_solving']: self.reset_algo_state()
                    else: self.start_visualization(); return
                # Speed control
                if hasattr(self, 'speed_buttons'):
                    for speed, rect in self.speed_buttons.items():
                        if rect.collidepoint(mouse_pos):
                            self.algo_animation_speed = speed
                            self.play_sound(self.sfx['click'])
                            return
                for algo, rect in self.algo_buttons.items():
                    if rect.collidepoint(mouse_pos) and not self.current_algo_state['is_solving']:
                        self.selected_algo = algo; self.reset_algo_state()

        if event.type == pygame.KEYDOWN:
            self.play_sound(self.sfx['click']) # Note: Pygame-web's behavior on first sound play may vary
            if self.mode == MODE_GAME:
                if time.time() < self.player_start_time + PLAYER_MOVE_DURATION: return
                if not self.game_won:
                    # Arrow keys for movement
                    if event.key == pygame.K_UP: self.move_player(-1, 0)
                    elif event.key == pygame.K_DOWN: self.move_player(1, 0)
                    elif event.key == pygame.K_LEFT: self.move_player(0, -1)
                    elif event.key == pygame.K_RIGHT: self.move_player(0, 1)
                if self.game_won and event.key == pygame.K_RETURN: self.load_level(self.current_level_index + 1)
            elif self.mode == MODE_INFO and event.key == pygame.K_ESCAPE: self.mode = MODE_ALGO

    # --- Game Loop (Now Asynchronous) ---

    async def run(self): # <--- CONVERTED TO ASYNC METHOD
        while self.running:
            for event in pygame.event.get():
                self.handle_input(event)

            self.update()

            self.screen.fill(self.colors['BG'])

            if self.mode == MODE_GAME:
                pygame.draw.rect(self.screen, self.colors['PATH'], (0, 0, MAZE_SIZE, MAZE_SIZE))
                self.draw_game_mode()
            elif self.mode == MODE_ALGO:
                self.draw_analysis_mode()
            elif self.mode == MODE_INFO:
                self.draw_info_mode()

            # Draw the extreme-right 'Maze Arina' strip after mode-specific content
            try:
                self.draw_arina_strip()
            except Exception:
                # Defensive: don't crash render loop if something unexpected happens drawing the strip
                pass

            pygame.display.flip()
            self.clock.tick(60)

            await asyncio.sleep(0) # <--- CRUCIAL: Yields control for web browser compatibility

        pygame.quit()

if __name__ == "__main__":
    app = MazeSolverApp()
    asyncio.run(app.run()) # <--- EXECUTES THE ASYNC RUN METHOD
