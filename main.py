import pygame
import random
import time
import math
import collections
import heapq
import asyncio

# --- 1. CONFIGURATION AND CONSTANTS ---

# Adjusted window size and layout
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 700
MAZE_SIZE = 700
ARINA_WIDTH = 160
UI_PANEL_WIDTH = SCREEN_WIDTH - MAZE_SIZE - ARINA_WIDTH

# Game/Visualization Modes (MODE_SPLASH is new)
MODE_SPLASH = -1
MODE_GAME = 0
MODE_ALGO = 1
MODE_INFO = 2

# Color Themes (omitted for brevity, assume previous definitions)
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
# Using DARK_COLORS as default
DARK_COLORS.update(LIGHT_COLORS) 
LIGHT_COLORS = DARK_COLORS 

PLAYER_THEMES = collections.OrderedDict([
    ("Dot (Classic)", {'draw_func': 'draw_dot', 'color_override': None}),
    ("Rocket", {'draw_func': 'draw_rocket', 'color_override': (255, 100, 100)}),
    ("Car", {'draw_func': 'draw_car', 'color_override': (100, 100, 255)}),
    ("Pikachu-Style", {'draw_func': 'draw_pikachu', 'color_override': (255, 255, 0)}),
])

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

LEVELS = [
    (7, "Level 1: Tiny (7x7)"),
    (10, "Level 2: Easy (10x10)"),
    (13, "Level 3: Moderate (13x13)"),
    (17, "Level 4: Challenging (17x17)"),
    (20, "Level 5: Hard (20x20)"),
    (23, "Level 6: Expert (23x23)"),
]

PLAYER_MOVE_DURATION = 0.2

# --- 2. MAZE CLASS AND GENERATION (Original, unchanged) ---

class Maze:
    """Manages the grid, walls, and maze generation."""
    def __init__(self, cols, rows):
        self.cols = cols
        self.rows = rows
        self.walls = collections.defaultdict(lambda: [True] * 4) 
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

# --- 3. ALGORITHM IMPLEMENTATIONS (Original, unchanged) ---

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
        # NOTE: WE REMOVED pygame.mixer.init() from here
        pygame.display.set_caption("Dual-Mode Interactive Maze Solver & Visualizer")
        
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.title_font = pygame.font.Font(None, 36)
        self.large_font = pygame.font.Font(None, 60) 

        self.current_theme = 'dark'; self.colors = DARK_COLORS

        # START THE APP IN SPLASH MODE
        self.mode = MODE_SPLASH 
        self.running = True; self.current_level_index = 0
        self.sound_enabled = True
        
        # Audio/SFX objects are initialized to None here
        self.sfx = None 

        # Game State (will be initialized after splash)
        self.player_r, self.player_c = (0, 0); self.game_start_time = time.time()
        self.time_elapsed = 0.0; self.game_won = False
        self.player_target_r, self.player_target_c = (0, 0); self.player_start_time = 0.0
        self.player_theme_index = 0
        self.animation_start = time.time(); self.bounce_duration = 0.5; self.float_duration = 1.0; self.bounce_height = 0.3; self.rotation_angle = 0; self.tail_offset = 0

        # Visualization State (will be initialized after splash)
        self.selected_algo = "BFS"; self.algo_generator = None
        self.algo_animation_speed = 5; self.algo_frame_counter = 0
        self.algo_metrics = {}; self.current_algo_state = self._reset_algo_state_dict()
        self.speed_levels = {1: "0.2x", 2: "0.5x", 5: "1.0x", 10: "2.0x", 20: "4.0x"}

        self.arina_enabled = True
        self.arina_buttons = {}
        try:
            self.arina_title_font = pygame.font.SysFont('georgia', 34, bold=True)
            self.arina_font = pygame.font.SysFont('georgia', 20)
        except Exception:
            self.arina_title_font = self.title_font
            self.arina_font = self.font
        self.arina_label_rect = None
        
    def initialize_game_state(self):
        """Called once the user clicks to start the game."""
        # *** CRITICAL CHANGE: INITIALIZE MIXER HERE ***
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            self.sfx = self._load_sfx()
        except Exception as e:
            # If mixer init fails (common in browsers), we continue without sound
            self.sound_enabled = False
            self.sfx = {}
            print(f"Warning: Audio initialization failed. Running without sound. Error: {e}")
        # *** END CRITICAL CHANGE ***

        self.load_level(self.current_level_index) # Now load the first maze
        self.mode = MODE_GAME # Switch to the default game mode

    def _load_sfx(self):
        # Original SFX loading logic (unchanged)
        sfx = {}
        sample_rate = 44100
        sfx['click'] = self._create_sine_wave(400, 0.08, sample_rate)
        sfx['move'] = self._create_sine_wave(262, 0.05, sample_rate)
        sfx['win'] = self._create_arpeggio([523, 659, 784, 1047], 0.1, sample_rate)
        return sfx

    def _create_sine_wave(self, frequency, duration, sample_rate):
        # Original wave creation logic (unchanged)
        n_samples = int(sample_rate * duration)
        sine_wave = [int(32767 * math.sin(2 * math.pi * frequency * i / sample_rate))
                     for i in range(n_samples)]
        sound_buffer = bytearray(2 * len(sine_wave))
        for i, val in enumerate(sine_wave):
            sound_buffer[2*i] = val & 0xFF
            sound_buffer[2*i + 1] = (val >> 8) & 0xFF
        return pygame.mixer.Sound(sound_buffer)

    def play_sound(self, sound):
        # Original sound play logic, now checks if sfx is loaded
        if self.sound_enabled and self.sfx:
            sound.play()
    
    # Rest of the original MazeSolverApp class (abbreviated here)
    def _create_arpeggio(self, frequencies, note_duration, sample_rate):
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
        return {
            'current': None, 'frontier': [], 'visited': set(), 'parent': {},
            'path': [], 'time': 0.0, 'steps': 0, 'path_len': 0,
            'is_solving': False, 'solved': False
        }

    def toggle_theme(self):
        if self.current_theme == 'dark':
            self.current_theme = 'light'
            self.colors = LIGHT_COLORS
        else:
            self.current_theme = 'dark'
            self.colors = DARK_COLORS
            
    def load_level(self, index):
        self.current_level_index = index % len(LEVELS)
        size, _ = LEVELS[self.current_level_index]
        self.maze = Maze(size, size)
        self.cell_size = MAZE_SIZE // size

        self.player_r, self.player_c = self.maze.start
        self.player_target_r, self.player_target_c = self.maze.start
        self.player_start_time = 0.0
        self.game_start_time = time.time()
        self.time_elapsed = 0.0
        self.game_won = False

        self.reset_algo_state(); self.algo_metrics = {}

    def reset_algo_state(self):
        self.algo_generator = None
        self.current_algo_state = self._reset_algo_state_dict()

    def draw_cell(self, r, c, color): pass # Implementation omitted for brevity
    def draw_maze(self): pass # Implementation omitted for brevity
    def draw_button(self, rect, text, is_active=False): pass # Implementation omitted for brevity
    def draw_player(self): pass # Implementation omitted for brevity
    def draw_dot(self, x, y, radius, color): pass # Implementation omitted for brevity
    def draw_rocket(self, x, y, radius, color): pass # Implementation omitted for brevity
    def draw_car(self, x, y, radius, color): pass # Implementation omitted for brevity
    def draw_pikachu(self, x, y, radius, color): pass # Implementation omitted for brevity
    
    def draw_splash_mode(self):
        self.screen.fill(self.colors['BG'])
        center_x = SCREEN_WIDTH // 2
        center_y = SCREEN_HEIGHT // 2
        
        # Title
        title_text = "MAZE SOLVER & VISUALIZER"
        title_surf = self.large_font.render(title_text, True, self.colors['SOLUTION'])
        title_rect = title_surf.get_rect(center=(center_x, center_y - 60))
        self.screen.blit(title_surf, title_rect)
        
        # Instructions
        instruction_text = "CLICK ANYWHERE TO START"
        instruction_surf = self.title_font.render(instruction_text, True, self.colors['TEXT_MAIN'])
        instruction_rect = instruction_surf.get_rect(center=(center_x, center_y + 20))
        self.screen.blit(instruction_surf, instruction_rect)

        # Pygbag/Web Note
        web_note_text = "Running on GitHub Pages via Pygbag (WebAssembly)"
        web_note_surf = self.font.render(web_note_text, True, self.colors['TEXT_SUB'])
        web_note_rect = web_note_surf.get_rect(center=(center_x, SCREEN_HEIGHT - 50))
        self.screen.blit(web_note_surf, web_note_rect)
        
        # Draw a bouncing cursor/indicator (Optional visual feedback)
        t = time.time() * 2 
        indicator_x = center_x
        indicator_y = center_y + 100 + math.sin(t) * 10
        pygame.draw.circle(self.screen, self.colors['PLAYER'], (int(indicator_x), int(indicator_y)), 10)


    def draw_game_mode(self): pass # Implementation omitted for brevity
    def draw_analysis_mode(self): pass # Implementation omitted for brevity
    def draw_info_mode(self): pass # Implementation omitted for brevity
    def draw_arina_strip(self): pass # Implementation omitted for brevity
    def move_player(self, dr, dc): pass # Implementation omitted for brevity
    def check_game_end(self): pass # Implementation omitted for brevity
    def start_visualization(self): pass # Implementation omitted for brevity
    def step_visualization(self): pass # Implementation omitted for brevity
    def end_visualization(self, parent_map, steps, found=True): pass # Implementation omitted for brevity

    def handle_input(self, event):
        if event.type == pygame.QUIT: self.running = False; return

        if self.mode == MODE_SPLASH:
            if event.type == pygame.MOUSEBUTTONDOWN or event.type == pygame.KEYDOWN:
                # We call play_sound ONLY if mixer init succeeded in initialize_game_state
                if self.sfx: self.play_sound(self.sfx['click'])
                self.initialize_game_state() 
                return

        # Placeholder input handling logic for other modes...
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = event.pos
            if self.sfx: self.play_sound(self.sfx['click'])

            if self.mode == MODE_INFO and hasattr(self, 'info_back_rect') and self.info_back_rect.collidepoint(mouse_pos): self.mode = MODE_ALGO; return
            if hasattr(self, 'sound_toggle_rect') and self.sound_toggle_rect.collidepoint(mouse_pos): self.sound_enabled = not self.sound_enabled; return
            if hasattr(self, 'theme_switch_rect') and self.theme_switch_rect.collidepoint(mouse_pos): self.toggle_theme(); return
        # ... and so on ...


    async def run(self):
        while self.running:
            for event in pygame.event.get():
                self.handle_input(event)

            # Update logic (original)
            if self.mode == MODE_GAME and not self.game_won and time.time() > self.player_start_time + PLAYER_MOVE_DURATION:
                self.player_r, self.player_c = self.player_target_r, self.player_target_c
                self.time_elapsed = time.time() - self.game_start_time
            elif self.mode == MODE_ALGO and self.current_algo_state['is_solving']:
                self.algo_frame_counter += 1
                if self.algo_frame_counter % (60 // self.algo_animation_speed) == 0:
                     self.step_visualization()

            self.screen.fill(self.colors['BG'])

            # --- RENDER BASED ON MODE ---
            if self.mode == MODE_SPLASH:
                self.draw_splash_mode()
            elif self.mode == MODE_GAME:
                pygame.draw.rect(self.screen, self.colors['PATH'], (0, 0, MAZE_SIZE, MAZE_SIZE))
                self.draw_game_mode() 
            elif self.mode == MODE_ALGO:
                self.draw_analysis_mode() 
            elif self.mode == MODE_INFO:
                self.draw_info_mode() 

            try:
                self.draw_arina_strip() 
            except Exception:
                pass

            pygame.display.flip()
            self.clock.tick(60)

            await asyncio.sleep(0) 

        pygame.quit()

if __name__ == "__main__":
    app = MazeSolverApp()
    asyncio.run(app.run())