import numpy as np


class RandomWalk:
    """Generate a synthetic series: a sinusoidal trend plus a random walk.

    Each step adds the current value of a sine wave (the *learnable* trend)
    plus a small random +/- fluctuation (the *irreducible* noise), accumulated
    over time. Randomness is isolated per instance via a NumPy Generator, so a
    given ``seed`` always reproduces the same series.
    """

    def __init__(self, seed=None,
                 starting_position=0,
                 max_step=1,
                 min_step=0,
                 period: int = 10,
                 amplitude: float = 5):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.starting_position = starting_position
        self.max_step = max_step
        self.min_step = min_step
        self.chain = [starting_position]
        self.step = 0

        # sinusoidal trend parametrisation: `period` full cycles over the series
        self.period = period
        self.amplitude = amplitude
        self.theta = 2.0 * np.pi * self.period

    def generate(self, n_steps):
        self.n_steps = n_steps
        t = np.arange(n_steps)
        self.sin_funct = self.amplitude * np.sin(t / n_steps * self.theta)
        while self.step < n_steps:
            self.chain.append(self.next_step(self.chain[-1]))
            self.step += 1
        return self.sin_funct

    def next_step(self, origin):
        # random direction (+1/-1) and magnitude -> a zero-drift random walk
        direction = 1 if self.rng.integers(0, 2) else -1
        fluctuation = self.rng.uniform(self.min_step, self.max_step)
        magnitude = self.sin_funct[self.step]
        return origin + magnitude + fluctuation * direction
