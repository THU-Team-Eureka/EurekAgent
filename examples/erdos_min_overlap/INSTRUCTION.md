## TASK
Write a program to solve this problem in the field of harmonic analysis, numerical optimization, and mathematical discovery.

Target: find h:[0,2]->[0,1] minimizing
  C5 = max_k integral h(x) * (1 - h(x+k)) dx

Discretization convention:
- represent h by n_points samples on [0,2)
- enforce 0 <= h[i] <= 1
- enforce sum(h) == n_points / 2 (equiv. integral of h is 1)
- objective in evaluator is lower C5 is better

Your run() function must return:
  (h_values, c5_bound, n_points)

Guidance:
- Keep runtime under budget_s in run()
- Prefer stable numerical procedures and valid outputs over aggressive but brittle search
- Return the best feasible solution found within time budget
- Smaller n_points (<1000) is preferred for speed
- You should write a program to solve the problem. Initial code is provided in the file `initial.py`. You can use it as a reference.
- To evaluate your program, we provide a evaluator in the file `evaluate.py`, which provides a final combined score. NEVER MODIFY THE EVALUATOR. Score can be obtained by `python evaluate.py --program_path initial.py --results_dir results`.

## HINTS and RECOMMENDATIONS
Recommendations:
You can use evolution algorithm to solve the problem. Keep a track of the solutions you have tried. And try to improve the solution. You can keep of promising solutions in a directory. And keep mutating, crossing, and evolving the solutions.
You can also keep a note in "NOTES.md" to record your thoughts and ideas, and improvements each iteration.

Note that the best known result is 0.38086945. Keep pushing to get a better result.