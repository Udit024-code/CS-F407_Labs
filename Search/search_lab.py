"""CS F407 - Search Lab: A* and BFS on the warehouse grid (pure Python, no ML libraries)."""
import heapq, math
from collections import deque

WAREHOUSE = """\
#################
#S....#.........#
#.###.#.#######.#
#...#.#.......#.#
###.#.#######.#.#
#...#.........#.#
#.###########.#.#
#.............#G#
#################"""

def parse(text):
    """Grid representation: list of strings. State = (row, col)."""
    grid = text.splitlines()
    start = goal = None
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch == 'S': start = (r, c)
            if ch == 'G': goal = (r, c)
    return grid, start, goal

MOVES = [('Up', -1, 0), ('Down', 1, 0), ('Left', 0, -1), ('Right', 0, 1)]

def successors(grid, state):
    """Transition function T: valid actions only (inside the grid and not an obstacle)."""
    r, c = state
    for name, dr, dc in MOVES:
        nr, nc = r + dr, c + dc
        if 0 <= nr < len(grid) and 0 <= nc < len(grid[nr]) and grid[nr][nc] != '#':
            yield name, (nr, nc)            # every move has cost 1

# ---- heuristics -------------------------------------------------------------
def manhattan(s, g): return abs(s[0] - g[0]) + abs(s[1] - g[1])
def euclidean(s, g): return math.hypot(s[0] - g[0], s[1] - g[1])
def zero(s, g):      return 0
def double_manhattan(s, g): return 2 * manhattan(s, g)

def rebuild(parent, state):
    path = []
    while state is not None:
        path.append(state); state = parent[state]
    return path[::-1]

def astar(grid, start, goal, h=manhattan):
    """A*: frontier = binary heap ordered by f = g + h; closed set prevents re-expansion."""
    frontier = [(h(start, goal), h(start, goal), 0, start)]   # entries: (f, h, g, state); ties -> smaller h
    parent, best_g, closed, expanded = {start: None}, {start: 0}, set(), 0
    while frontier:
        f, _, g, s = heapq.heappop(frontier)      # lowest f is expanded next
        if s in closed: continue                  # stale duplicate entry
        closed.add(s); expanded += 1
        if s == goal:                             # goal test on expansion
            path = rebuild(parent, s)
            return dict(found=True, path=path, length=len(path) - 1, expanded=expanded)
        for _, n in successors(grid, s):
            ng = g + 1
            if n not in best_g or ng < best_g[n]:
                best_g[n] = ng; parent[n] = s
                heapq.heappush(frontier, (ng + h(n, goal), h(n, goal), ng, n))   # f(n) = g(n) + h(n)
    return dict(found=False, path=None, length=None, expanded=expanded)

def bfs(grid, start, goal):
    """BFS: FIFO queue, same goal test and same 'expanded' definition as astar()."""
    frontier, parent, closed, expanded = deque([start]), {start: None}, set(), 0
    while frontier:
        s = frontier.popleft()
        if s in closed: continue
        closed.add(s); expanded += 1
        if s == goal:
            path = rebuild(parent, s)
            return dict(found=True, path=path, length=len(path) - 1, expanded=expanded)
        for _, n in successors(grid, s):
            if n not in parent:
                parent[n] = s; frontier.append(n)
    return dict(found=False, path=None, length=None, expanded=expanded)

def show(grid, path):
    g = [list(r) for r in grid]
    for (r, c) in (path or [])[1:-1]: g[r][c] = '*'
    return "\n".join("".join(r) for r in g)

def valid_path(grid, path, start, goal):
    """Independent validator: starts at S, ends at G, unit steps only, no obstacles."""
    if path[0] != start or path[-1] != goal: return False
    for a, b in zip(path, path[1:]):
        if abs(a[0]-b[0]) + abs(a[1]-b[1]) != 1 or grid[b[0]][b[1]] == '#': return False
    return True

if __name__ == "__main__":
    grid, s, g = parse(WAREHOUSE)
    res = astar(grid, s, g)
    print(show(grid, res['path'])); print(res['length'], res['expanded'])
