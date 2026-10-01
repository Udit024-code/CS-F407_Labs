"""Tests for the A* / BFS warehouse agent (run: python3 search_tests.py)."""
from search_lab import *
T2 = "#####\n#SG##\n#####"
T3 = "#######\n#S....#\n###.###\n#...#G#\n#######"
T4a = "#####\n#S..#\n#.#.#\n#..G#\n#####"                 # two equal-length shortest paths
T4b = "#######\n#S.G..#\n#.###.#\n#.....#\n#######"          # short path and a long detour
OPEN = "#########\n#S......#\n#.......#\n#.......#\n#......G#\n#########"
TRAP = open('trap_map.txt').read() if __import__('os').path.exists('trap_map.txt') else None
def run(text, algo, **k):
    grid, s, g = parse(text)
    r = algo(grid, s, g, **k) if algo is astar else algo(grid, s, g)
    r['valid'] = valid_path(grid, r['path'], s, g) if r['found'] else None
    return grid, r
if __name__ == "__main__":
    for n, t in [("Test 1 warehouse", WAREHOUSE), ("Test 2 trivial", T2), ("Test 3 no solution", T3), ("Test 4a", T4a), ("Test 4b", T4b)]:
        for nm, algo in [("A*", astar), ("BFS", bfs)]:
            grid, r = run(t, algo)
            print(f"{n:20s} {nm:4s} found={r['found']} length={r['length']} expanded={r['expanded']} valid={r['valid']}")
