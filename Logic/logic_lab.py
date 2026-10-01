"""CS F407 - Logical Planning Lab: STRIPS-style planner with breadth-first search."""
from collections import deque
from dataclasses import dataclass

@dataclass(frozen=True)
class Action:
    name: str
    pos_pre: frozenset = frozenset()   # facts that must be in the state
    neg_pre: frozenset = frozenset()   # facts that must NOT be in the state
    pos_eff: frozenset = frozenset()   # facts added
    neg_eff: frozenset = frozenset()   # facts removed

def A(name, pos_pre=(), neg_pre=(), pos_eff=(), neg_eff=()):
    return Action(name, frozenset(pos_pre), frozenset(neg_pre), frozenset(pos_eff), frozenset(neg_eff))

def applicable(state, a):
    """S |= Preconditions(a): all positive preconditions hold and no negative precondition holds."""
    return a.pos_pre <= state and not (a.neg_pre & state)

def apply(state, a):
    """S' = (S - neg_eff) + pos_eff"""
    return frozenset((state - a.neg_eff) | a.pos_eff)

def bfs_plan(initial, goal, actions):
    """Breadth-first search over states. Returns (plan, states) or (None, None) if no plan exists."""
    initial, goal = frozenset(initial), frozenset(goal)
    frontier, parent = deque([initial]), {initial: None}
    while frontier:
        s = frontier.popleft()
        if goal <= s:                                   # goal test: every goal fact holds
            plan, states = [], [s]
            while parent[s] is not None:
                s, a = parent[s]; plan.append(a); states.append(s)
            return plan[::-1], states[::-1]
        for a in actions:
            if applicable(s, a):                        # logic: is the action possible here?
                n = apply(s, a)                         # effects: successor state
                if n not in parent:
                    parent[n] = (s, a); frontier.append(n)   # search: explore alternatives
    return None, None

LOCS = ['A', 'B', 'C']
LINKS = [('A', 'B'), ('B', 'A'), ('B', 'C'), ('C', 'B')]

def warehouse_actions(with_pickup=True):
    acts = []
    for x, y in LINKS:
        acts.append(A(f"Move({x},{y})", [f"At(Robot,{x})"], (), [f"At(Robot,{y})"], [f"At(Robot,{x})"]))
    for l in LOCS:
        if with_pickup:
            acts.append(A(f"PickUp(Package,{l})", [f"At(Robot,{l})", f"At(Package,{l})"], (), ["Holding(Package)"], [f"At(Package,{l})"]))
        acts.append(A(f"Drop(Package,{l})", [f"At(Robot,{l})", "Holding(Package)"], (), [f"At(Package,{l})"], ["Holding(Package)"]))
    return acts

INITIAL = {"At(Robot,A)", "At(Package,A)"}
GOAL = {"At(Package,C)"}

def independent_check(initial, goal, plan):
    """Re-verify a plan step by step, with its own bookkeeping (does not call bfs_plan)."""
    s, log = set(initial), []
    for a in plan:
        ok = all(p in s for p in a.pos_pre) and all(n not in s for n in a.neg_pre)
        if not ok: return False, log
        s -= a.neg_eff; s |= a.pos_eff; log.append((a.name, sorted(s)))
    return set(goal) <= s, log

if __name__ == "__main__":
    plan, states = bfs_plan(INITIAL, GOAL, warehouse_actions())
    print([a.name for a in plan])
    for st in states: print(sorted(st))
