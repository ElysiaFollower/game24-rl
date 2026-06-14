# Use multiset-isolated train, validation, and test splits

We will split solvable 24-point puzzles by sorted multiset, not by generated trace rows, and keep independent out-of-distribution evaluation splits for hard ToT puzzles, Countdown-style targets, and unsolvable puzzles. This protects the visible score from memorization, makes the comparison more persuasive, and keeps external baselines meaningful because every score is tied to an explicit evaluation setting.
