# Random_Walker_2d
 Supports Lattice, Physical and Continuous Time Random Walks.

 The logic is as follows:
 1. Instantiate a ```RandomWalker2D``` (or related) object. This contains the physics of the walk, i.e. only lattice steps are allowed or a physical random walk with general 2D-coordinates.
 2. Instantiate a ```RandomWalk``` object. This object performs the random walk itself. It can also handle multiple walkers, branching and annihilation phenomena.

 # To Do list
 - [] Write base class ```RandomWalker``` and inherit to ```RandomWalker2D``` and ```LatticeWalker2D```
 - [] Write base class ```RandomWalk``` and inherit to the others.
