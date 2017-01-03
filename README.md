# DOF_estimation (alpha version)
Degrees of Freedom estimation for climate data

variables needs to have dimension (T,X,Y)

Spatial DOF:

USAGE: spatial_dof = B_method(var,S=3000,estimates=100) # can be slow, just reduce estimates to a lower number

Temporal DOF (Effective Sample Size): 

USAGE: ratio = ESS(var1,var2) # var1 can be the same as var2
 
COMBINE: DOF = spatial_dof.mean() * ratio.mean() * timesteps 
