# Random_Walker_2d
 Supports Lattice, Physical and Continuous Time Random Walks.

 This code has been used in the publication:
 
 *The diffusion metrics of African swine fever in wild boar*, Lentz, H.H.K., Bergmann, H., Conraths, F.J. et al. Sci Rep 13, 15110 (2023).
 https://doi.org/10.1038/s41598-023-42300-0

## Initiation
 The logic is as follows:
 1. Instantiate a ```RandomWalker2D``` (or related) object. This contains the physics of the walk, i.e. only lattice steps are allowed or a physical random walk with general 2D-coordinates.
 2. Instantiate a ```RandomWalk``` object. This object performs the random walk itself. It can also handle multiple walkers, branching and annihilation processes.
