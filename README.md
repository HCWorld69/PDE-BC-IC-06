# PDE-BC-IC-06
PDE Pseudocode Blood fluid flow differential equation
a pseudocode python linear neural network code to solve following boundary condition PDE equation using pytorch in python and solving for 'u' , and assuming values when needed: $$ \frac{\partial u_z}{\partial t}=v \frac{1}{r} \frac{\partial}{\partial r}\left(r \frac{\partial u_z}{\partial r}\right)-\frac{1}{\rho} \frac{\partial p}{\partial z} $$ where $v$ is the kinematic viscosity and $\rho$ is the density of the fluid. The position along the radius of the blood vessel is measured by $r$. The pressure gradient oscillates in time with frequency, $\omega(\mathrm{rad} / \mathrm{s})$, to simulate the pumping action of the heart: $$ -\frac{\partial p}{\partial z}=\frac{\Delta p}{L} \cos (\omega t) $$ The initial and boundary conditions for this problem are as follows: Initial condition: $$ t=0 \quad u_z=u_{z 0} $$ Boundary conditions: $\quad t>0 \quad r=0 \quad \frac{\partial u_z}{\partial r}=0 \quad u_z$ is finite $$ r=R \quad u_z=0 $$
