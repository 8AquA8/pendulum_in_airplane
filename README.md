# pendulum_in_airplane

Lets imagine a box with a pendulum is hung to the ceiling of the airplane room.
A pivot of the box adjusts the orientation of the box relative to the airplane direction so that the angular momentum of the box-plane system is conserved.
A pendulum is placed such that one end is fixed and the other end has a ball of mass M with massless rigid rod.
The airplane can change its manueuver to create different values of gravity, for example, near 0G.

Let's simulate the following situation by solving the equation of motion via Runge-Kutta method.

a) The metal pendulum is initially at rest.

b) Apply periodic external force to move the pendulum.

c) The mass of the pendulum reaches 60 degree relative to its equilibrium position.

d) You remove the periodic external force and the airplane lifts off.

e) The hyper gravity due to airplane movement is shown in ipynb file.

f) Eventually the pendulum stops due to the air drag and frictions.



I recommend you to see the pendulm_sim_plots.ipynb file first to see the whole algorithm flow of the simulation.
pendulum_sim_video.py will make you a video of how the pendulum motions over time.
