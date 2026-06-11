## TASK
Write a program to solve this problem in the field of harmonic analysis, numerical optimization, and mathematical discovery.

Target: find a nonnegative coefficient sequence `a` minimizing
  2*n*max(convolve(a,a)) / (sum(a)^2)

where n = len(a) and convolve(a,a) is the full discrete self-convolution.

Contract:
- Implement `run(seed, budget_s, **kwargs)` returning `list[float]`
- Keep all values finite and nonnegative
- Values are clipped to [0, 1000]; sum must be >= 0.01
- Use budget_s as a hard runtime budget and return the best sequence found
- Lower is better
- You should write a program to solve the problem. Initial code is provided in the file `initial.py`. You can use it as a reference.
- To evaluate your program, we provide a evaluator in the file `evaluate.py`, which provides a final combined score. NEVER MODIFY THE EVALUATOR. Score can be obtained by `python evaluate.py --program_path initial.py --results_dir results`.

## HINTS and RECOMMENDATIONS
Recommendations:
You can use evolution algorithm to solve the problem. Keep a track of the solutions you have tried. And try to improve the solution. You can keep of promising solutions in a directory. And keep mutating, crossing, and evolving the solutions.
You can also keep a note in "NOTES.md" to record your thoughts and ideas, and improvements each iteration.
It's recommended to save important iterations (program) and keep them in iterations/ directory (as program_v*.py).

Note that the best known result is 1.5028628969. Keep pushing to get a better result.