% planner.pl - warehouse knowledge base used as an independent verifier (CS F407 logic lab)

% --- Task 6: facts and a rule -------------------------------------------
connected(a,b).
connected(b,a).
connected(b,c).
connected(c,b).

can_move(X,Y) :- connected(X,Y).

% --- Task 7: checking proposed moves ------------------------------------
valid_move(X,Y) :- connected(X,Y).

% (extra) check a whole route, e.g. valid_route([a,b,c]).
valid_route([_]).
valid_route([X,Y|T]) :- valid_move(X,Y), valid_route([Y|T]).

% --- Task 8: chained rules ----------------------------------------------
wet_road.
slippery :- wet_road.
reduce_speed :- slippery.
