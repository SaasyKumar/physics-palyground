# Black Hole (Consideration for simulation)
- Gravitational constant is `0.1` and mass of black hole is `1.0`. so Force act on each particle
$$
\frac{GM}{r^2} = \frac{0.15}{r^2}
$$
- The black hole is positioned in (0.5,0.5) center of the cartesian coords of simulation. Even horizon radius is `0.025`.That is the particles which are inside the horizon radius will be absorbed by the black hole.[0.475,0.525] is the range of the horizon radius.

# Particles
- 70% is orbiting particles which will be spawned from left [0.1,0.45] and right [0.55,0.85]; outside horizon radius. 
- 30% free particles from [0.0,0.1] and [0.85,1.0] and move towards the center
- the position, velocity and color of each particles are stored in a vector.
- Every particle is spawned in certain random angle with the black hole and these polar coordinates are use to calculate the initial position and velocity of the particles. The angle is generated randomly between 0 to 2π radians. since the black hole is at the center of the simulation, the polar coordinates are converted to cartesian coordinates using the following formulas:
$$
x = r \cdot \cos(\theta) + x_{bh}
$$
$$
y = r \cdot \sin(\theta) + y_{bh}
$$
### Orbiting 70
- since the particles are orbiting the black hole. The the gravitational force will be equal to the centripetal force. The centripetal force is given by the formula:
$$
F_c = \frac{mv^2}{r}
$$
- The velocity of the orbiting particles is calculated using the formula:
$$
v = \sqrt{\frac{GM}{r}}
$$
- substituting the values of G and M, we get:
$$
v = \sqrt{\frac{0.15}{r}}
$$
- r ranges from [0.05,0.4] so the velocity ranges from [1.73,0.61]

### Free 30
- standard r of 0.5
- so the velocity need for orbiting is 0.55. So th range for velocity is set from [0.1,0.3] so that the simulation is observable and the particles are non orbiting
- the particles target random parts of center rather than the center [0.5,0.5] by using an offset


# Update
- At each moment of the simulation like each frame or second the stationary particles(ref to the previous timeframe) affected by the gravitation of black hole and the particle's next position and velocity is calculated
- Then it is updated in vector storage and also in simulation