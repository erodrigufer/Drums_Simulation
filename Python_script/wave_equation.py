'''
LDV Python Course -- Numerics: Wave Equation Solver
SS 2019
'''

import numpy as np
print('numpy: '+np.version.full_version)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

#import matplotlib
#print('matplotlib: '+matplotlib.__version__)
from initialConditions import gaussianDistribution

import math

# ------------------------------------------------------
# (1) Declare physical properties of the drum membrane

meshDimension = 40  # Dimension of mesh grid in each direction

rho = 0.295  # Speed of wave propagation
# rho determines the pitch

eta = 2e-4  # damping coefficient
# --> drum heads have a damping coefficient near to 0 => they are almost lossless

# General user input

sampleFrequency = 44.1e3  # 48kHz Sampling frequency, more than double than
# human hearing range

audioLength = 0.5  # in seconds

# ------------------------------------------------------
# (2) Initialize time scale and audio signal

# Check documentation for numpy.arange
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.arange.html#numpy.arange

t = np.arange(0, audioLength, 1/sampleFrequency)
# start audio time at 0 seconds, run through to audioLength
# sampling at the sampleFrequency

fps = 1000  # frame per sec
frn = t.size  # frame number of the animation
# number of frame equals as the number of time steps

# initialize array for audio output with zeros => must be
# same size as t array

audio = np.zeros(t.size)

# t.size is an OOP method which gives out the size of an instance
# (array in this case) it gives out its length

# ------------------------------------------------------
# (3) Initialize matrices for field at times t, t-1, t-2 and for
# input strike

z = np.zeros((meshDimension,meshDimension,t.size))  # current values of z displacement of
#  membrane grid
z_t1 = np.zeros((meshDimension,meshDimension))  # z displacement of membrane at time
#  t-1
z_t2 = np.zeros((meshDimension,meshDimension))  # z displacement of membrane at time
#  t-2

# Initialize the matrix that will later be filled with the initial
# conditions of the first strike on the membrane (gaussian distribution)

z_initial_conditions = np.zeros((meshDimension,meshDimension))

# ------------------------------------------------------
# (4) Initialize matrix for input strike with a gaussian distribution

amplitude = 0.8 # should be positive, determines height of input strike
leftBoundary = -1 # at what point in the x/y axis does the grid start
rightBoundary = 1 # at what point in the x/y axis does the grid end

x = np.linspace(leftBoundary, rightBoundary, meshDimension)
y = np.linspace(leftBoundary, rightBoundary, meshDimension)
X, Y = np.meshgrid(x, y) # create X,Y matrices with points in x/y space
# where equation will be solved

#  set position of the strike in the x, y axis
#  leftboundary < x,y_strike < rightBoundary
x_strike = x[(math.floor(x.size/2))]
y_strike = y[(math.floor(y.size/2))]
# use floor function to get an integer

standardDeviation = -40
#  the standard deviation will influence how wide
#  the gauss distribution will be
#  the boundary conditions of the border never being displaced
#  vertically (z-axis) must always be met

#  the function np.power performs element-wise the power of something
#  the function np.exp computes the exponential value also element-wise
#  for all the elements of a matrix or an array

z_initial_conditions = gaussianDistribution(meshDimension,standardDeviation,amplitude,x_strike,y_strike,leftBoundary,rightBoundary)
#TODO guarantee that the boundary conditions are fullfilled -> the boundary should be always 0
# !! ENFORCE BOUNDARY CONDITIONS!!


# ------------------------------------------------------
# (5) Solve the difference equation for all time-steps

# Before for-loop starts z must be the initial conditions
z[:,:,0] = z_initial_conditions
z_t1 = z_initial_conditions
z_t2 = z_initial_conditions

# Define the point in the grid which will be used to create the audio output
# the current vertical displacement z, which will create the audio wave
# is one single specific point in the grid, in reality it is much more complicated
# probably. It is the addition of many waves coming from all points in the membrane at once
#audio_output_coordinates = math.floor((z.size)/2) # use the geometrical mean
audio_output_coordinates = 25
# from both boundaries

n = meshDimension # use n to shorten the difference expression below

for i in range(t.size): # start a loop from 0 to last element of t

    audio[i] = z[(audio_output_coordinates,audio_output_coordinates,i)]
    # copy the current z value at the audio output coordinates to the time i in the audio output

    z[2:n-1,2:n-1,i] = 1/(1+eta)*(rho *(z_t1[3:n, 2:n-1] + z_t1[
1:n-2,2:n-1]+z_t1[2:n- 1,3:n]+z_t1[
2:n-1,1:n-2]-4 * z_t1[2: n - 1,2: n- 1])+ 2* z_t1[2:n-1,2:n-1]-(1-eta)*(z_t2[2:n-1,2:n-1]))

    # TODO enforce boundary conditions here as well -> borders should be 0


    # update z at time t-1 and z at time t-2
    z_t2 = z_t1 # the value at t-2 is now what the value at t-1 was
    z_t1 = z[:,:,i] # in the next time step the actual z value will be in the past

# ------------------------------------------------------
# (6) Animation for z

def update_plot(frame_number, zarray, plot):
    plot[0].remove()
    plot[0] = ax.plot_surface(X, Y, zarray[:, :, frame_number], cmap="viridis")


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plot = [ax.plot_surface(X, Y, z[:, :, 0], color='1', rstride=1, cstride=1)]
ax.set_zlim(0, 1.1)
ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(z, plot), interval=1000 / fps)

# -> fig : matplotlib.figure.Figure
# The figure object that is used to get draw, resize, and any other needed events.
# -> interval : number, optional
# Delay between frames in milliseconds. Defaults to 200.
# -> frames : iterable, int, generator function, or None, optional
# Source of data to pass func and each frame of the animation
# In all of these cases, the values in frames is simply passed
# through to the user-supplied func and thus can be of any type.
# In this case update_plot
#ani.save('dynamic_images2.html')


plt.show()











