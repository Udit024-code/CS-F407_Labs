"""CS F407 - Agents Lab: a goal-based warehouse navigation agent (BFS search)."""
from collections import deque

WAREHOUSE = """\
#####################
#S....#............G#
#.##....##########..#
#....##.............#
#.######.###.#.###..#
#........#..........#
#####################"""

MOVES = {'Up': (-1, 0), 'Down': (1, 0), 'Left': (0, -1), 'Right': (0, 1)}

class Environment:
    """The warehouse: a 2-D grid. Obstacles '#' cannot be crossed."""
    def __init__(self, text):
        self.grid = [list(r) for r in text.splitlines()]
        for r, row in enumerate(self.grid):
            for c, ch in enumerate(row):
                if ch == 'S': self.start = (r, c)
                if ch == 'G': self.goal = (r, c)
    def free(self, pos):
        r, c = pos
        return 0 <= r < len(self.grid) and 0 <= c < len(self.grid[r]) and self.grid[r][c] != '#'
    def neighbours(self, pos):
        for name, (dr, dc) in MOVES.items():
            nxt = (pos[0] + dr, pos[1] + dc)
            if self.free(nxt): yield name, nxt

class GoalBasedAgent:
    """Percept -> update state -> decide (search for a path to the goal) -> act."""
    def __init__(self, env):
        self.env, self.state, self.goal = env, env.start, env.goal   # current state and explicit goal
    def goal_test(self, state): return state == self.goal
    def plan(self):
        """Decision-making component: breadth-first search from the current state to the goal.
        BFS is complete and returns a shortest path because every move has the same cost."""
        frontier, parent, expanded = deque([(self.state, None, None)]), {self.state: None}, 0
        came = {self.state: (None, None)}
        while frontier:
            s, _, _ = frontier.popleft(); expanded += 1
            if self.goal_test(s):
                actions, path = [], [s]
                while came[s][0] is not None:
                    prev, act = came[s]; actions.append(act); path.append(prev); s = prev
                return actions[::-1], path[::-1], expanded
            for act, n in self.env.neighbours(s):
                if n not in came: came[n] = (s, act); frontier.append((n, s, act))
        return None, None, expanded          # no path exists
    def act(self, action):
        dr, dc = MOVES[action]; self.state = (self.state[0] + dr, self.state[1] + dc)

def draw(env, path):
    g = [row[:] for row in env.grid]
    for (r, c) in path[1:-1]: g[r][c] = '*'
    return "\n".join("".join(r) for r in g)

def run(text=WAREHOUSE):
    env = Environment(text); agent = GoalBasedAgent(env)
    actions, path, expanded = agent.plan()
    if actions is None: return dict(found=False, expanded=expanded)
    for a in actions: agent.act(a)                       # execute the plan
    assert agent.goal_test(agent.state)
    return dict(found=True, actions=actions, path=path, expanded=expanded, env=env)

if __name__ == "__main__":
    r = run()
    if r['found']:
        print(draw(r['env'], r['path'])); print("moves:", len(r['actions']), "expanded:", r['expanded']); print(" ".join(r['actions']))
    else: print("No path exists")
    print("No-path test:", run("#####\n#S#G#\n#####")['found'])
