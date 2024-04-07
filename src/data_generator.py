#import random_walker as rw
import src.random_walker as rw

class Data(rw.RandomWalk):
    def __init__(self,
                 n_steps = 1000,
                 n_training_steps = 900,
                 seed=0,
                 starting_position=0,
                 max_step=1,
                 min_step=0,
                 period:int=10,
                 amplitude:float=5
                 ):
         super().__init__(seed,
                 starting_position,
                 max_step,
                 min_step,
                 period,
                 amplitude)
         
         self.n_steps = n_steps
         self.n_training_steps = n_training_steps
         self.x_dataset = []
         self.y_dataset = []

    def generate_datasets(self,n_datasets,seed=None):
        for i in range(0,n_datasets):
            if not seed:
                rdm_walk = rw.RandomWalk(i,
                                         self.starting_position,
                                         self.max_step,
                                         self.min_step,
                                         self.period,
                                         self.amplitude)
            else:
                rdm_walk = rw.RandomWalk(seed,
                                         self.starting_position,
                                         self.max_step,
                                         self.min_step,
                                         self.period,
                                         self.amplitude)
            self.sin_funct = rdm_walk.generate(self.n_steps)
            self.x_dataset.append(rdm_walk.chain[:self.n_training_steps])
            self.y_dataset.append(rdm_walk.chain[self.n_training_steps:])
        return self.x_dataset, self.y_dataset 