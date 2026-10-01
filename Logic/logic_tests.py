"""Tests A, B, C for the planner (run: python3 logic_tests.py)."""
from logic_lab import *

def fmt(plan): return "No plan found" if plan is None else " -> ".join(a.name for a in plan)

def testA():
    plan, states = bfs_plan(INITIAL, GOAL, warehouse_actions())
    ok, log = independent_check(INITIAL, GOAL, plan)
    return plan, states, ok, log

def testB():                      # PickUp removed
    plan, _ = bfs_plan(INITIAL, GOAL, warehouse_actions(with_pickup=False)); return plan

def testC():
    acts = warehouse_actions(with_pickup=False)
    p_pkg, _ = bfs_plan(INITIAL, {"At(Package,C)"}, acts)        # package goal, robot can reach C
    p_rob, _ = bfs_plan(INITIAL, {"At(Robot,C)"}, acts)          # robot goal
    decoy = A("Wander(A,B)", ["At(Robot,A)"], (), ["At(Robot,B)"], ["At(Robot,A)"])   # moves robot only
    p_dec, _ = bfs_plan(INITIAL, GOAL, warehouse_actions() + [decoy])
    return p_pkg, p_rob, p_dec

def applicability():
    acts = {a.name: a for a in warehouse_actions()}
    s = frozenset(INITIAL)
    return {n: applicable(s, acts[n]) for n in ["PickUp(Package,A)", "PickUp(Package,B)", "Drop(Package,C)", "Move(A,B)", "Move(A,C)" if "Move(A,C)" in acts else "Move(B,C)"]}

if __name__ == "__main__":
    plan, states, ok, log = testA(); print("A:", fmt(plan), "valid:", ok)
    print("B:", fmt(testB()))
    pk, rb, dc = testC(); print("C package goal:", fmt(pk)); print("C robot goal:", fmt(rb)); print("C decoy:", fmt(dc))
    print(applicability())
