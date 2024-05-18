from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from math import *
from numpy.linalg import inv
import pygame
import sys
import cv2

# Initialize Pygame
pygame.init()

# Setup for text rendering
font_size = 20
font = pygame.font.Font(None, font_size)

def render_text(screen, text, position, color=(0, 0, 0)):
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, position)

def objective(params):
    A, B, C, D = params
    
    # Define the sine function
    f = lambda t: A * np.sin(B * t + C) + D
    
    # Errors at each given point
    error1 = f(20) - 9.8
    error2 = f(30) - (1.6 * 9.8)
    error3 = f(40) - 9.8
    error4 = f(42) - 0.0
    
    # Sum of squared errors
    return 10*error1**2 + error2**2 + error3**2 + 15*error4**2

def fitted_sine_function(A, B ,C ,D, t):
    return A * np.sin(B * t + C) + D

def point(angle):
    x = l * sin(angle) + x_center
    y = l * cos(angle) + y_center
    return (x, y)

def render (posxy):
    screen.fill(white)
    pygame.draw.line(screen, black, (x_center, y_center), (posxy[0], posxy[1]), 2)


    pygame.draw.line(screen, grey, (x_center, y_center), (x_center+l*(sqrt(3.0))/2.0, y_center+l/2.0), 2)
    pygame.draw.line(screen, grey, (x_center, y_center), (x_center-l*(sqrt(3.0))/2.0, y_center+l/2.0), 2)
    
    pygame.draw.circle(screen, green, (posxy[0], posxy[1]), 10)
    
def G(t, y, df):
    F[0] = df-g*sin(y[1]) - c*y[0]
    F[1] = y[0]
    return inv_L.dot(F)

def RK4(t, y, delta_t, df):
    k1 = G(t, y, df)
    k2 = G(t+0.5*delta_t, y+0.5*delta_t*k1, df)
    k3 = G(t+0.5*delta_t, y+0.5*delta_t*k2, df)
    k4 = G(t+1.0*delta_t, y+1.0*delta_t*k3, df)
    return k1/6.0 + 2.0*k2/6.0 + 2.0*k3/6.0 + k4/6.0

initial_guess = [10, 0.1, 0, 9.8]

#canvas size
width = 800
height = 800

fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter('hyper_gravity_pendulum.avi', fourcc, 100.0, (width, height))

screen = pygame.display.set_mode((width, height))

#predefined colors
green = pygame.Color('green')
white = pygame.Color('white')
black = pygame.Color('black')
grey = pygame.Color('grey')

# fill the screen with white color
screen.fill(white)

# update the display
pygame.display.update()

#clock for frames
clock = pygame.time.Clock()

l = 300 # length in pixel

# location of the origin
x_center = width*0.5
y_center = height*0.5

g = 9.8 

result = minimize(objective, initial_guess)
param = result.x

A, B, C, D = param

time_lift = np.linspace(20, 42, 1000)
time_drop = np.linspace(78, 100, 1000)

# Calculate the sine values for the time range
sin_drop = fitted_sine_function(A, B, C, D, 120-time_drop)
sin_lift = fitted_sine_function(A, B, C, D, time_lift)

hyperG = np.zeros(4000)
hyperG[: len(sin_lift)] = sin_lift  # First 22 entries from sin_drop
hyperG[len(sin_lift) + int(36/22*1000): len(sin_lift) + int(36/22*1000)+len(sin_drop)] = sin_drop  # Last part of the array from sin_lift
hyperG[len(sin_lift) + int(36/22*1000)+len(sin_drop):] = 9.8

ll = 13.5 # length 13.5 meters
c = 0.1 # damping term due to friction
F0 = .65 # driving force amplitude
omega = sqrt(g/ll)

t = 0.0
delta_t = 0.02

#init condition
theta0 = 0.0*pi/180.
thetadot0 = 0.0
y = np.array([thetadot0, theta0])

L = np.array([[ll, 0.0],
              [0.0, 1.0]])

F = np.array([0.0, 0.0])

inv_L = inv(L)
dF = 0

count = 0
lift_start = False
lift_time = 0.0
hypertime = 0
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            video.release()
            pygame.quit()
            sys.exit()
            
    # updating the positions
    xy = point(y[1])
    
    render(xy)
    
    # Render gravity and time text
    gravity_text = f"Gravity: {g:.2f} m/s²"
    time_text = f"Time: {t:.2f} s"
    render_text(screen, gravity_text, (50, 50), black)
    render_text(screen, time_text, (50, 80), black) 
    
    t += delta_t
    
    if lift_start == False:
        dF=F0*np.cos(omega*t)

    if (pi/3 - 0.01) < abs(y[1]) < (pi/3 + 0.01) and -0.01 < abs(y[0]) < 0.01 and lift_start==False:
        lift_time = t
        lift_start = True
        dF = 0
        print(f"Lift starts at t={lift_time:.2f} seconds")
        
        
    if lift_start:
        if hypertime < len(hyperG):
            g = hyperG[hypertime]
            hypertime += 1  
        else:
            g = 9.8
            
    if lift_time+300<=t:
        break
        
    y = y + delta_t * RK4(t, y, delta_t, dF)
    
    frame = pygame.surfarray.array3d(pygame.display.get_surface())
    frame = frame.transpose([1, 0, 2])
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 
    
    video.write(frame)
    
    clock.tick(60)
    pygame.display.update()
    
