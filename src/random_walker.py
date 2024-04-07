import numpy as np
import numpy.random as rdm

class RandomWalk():
    def __init__(self,seed=0,
                 starting_position=0,
                 max_step=1,
                 min_step=0,
                 period:int=10,
                 amplitude:float=5):
        self.seed = seed
        if self.seed:
            rdm.seed(seed)
        self.starting_position = starting_position
        self.max_step = max_step
        self.min_step = min_step
        self.chain = [self.starting_position]
        self.step = 0

        #sinusodial trend parametrisation
        self.period = period
        self.amplitude = amplitude

        self.theta = 2.0 * np.pi * self.period
        
    def generate(self,n_steps):
        self.n_steps = n_steps
        self.sin_funct = self.amplitude * np.sin([i * 1/self.n_steps * self.theta for i in list(range(0,self.n_steps))])
        while self.step < self.n_steps:
            self.chain+=[self.next_step(self.chain[-1])]
            self.step+=1
        return self.sin_funct

    def next_step(self,origin):
        # if true then positive direction else negative
        if rdm.random_integers(0,1):
            direction=1
        else:
            direction=-1
        
        # size of step dependent on min and max step size
        fluctuation = rdm.uniform(self.min_step,self.max_step) 
        magnitude  = self.sin_funct[self.step]
        
        return origin + magnitude + fluctuation*direction
        
