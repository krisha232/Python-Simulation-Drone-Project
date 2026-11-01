import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp #solve equation
from scipy.optimize import minimize #optimize

m, g, rho = 1.5, 9.81, 1.225
Cd, A_body, A_rotor = 1.1, 0.05, 0.3
wind = np.array([4.0, 0.0]) #random
target = np.array([80.0, 0.0]) #random

def Phover(alpha):
    T = m*g/np.cos(alpha)
    return (T**1.5)/np.sqrt(2*rho*A_rotor)

def dragF(v_rel):
    vmag = np.linalg.norm(v_rel)
    if vmag == 0:
        return np.zeros(2)
    return 0.5*rho*Cd*A_body*vmag*v_rel
     
def dynamics(t, state, u):
    x, y, vx, vy = state
    v = np.array([vx, vy])
    theta1, alpha1, theta2, alpha2 = u

    # two sections
    if t < 5:
        theta, alpha = theta1, alpha1
    else:
        theta, alpha = theta2, alpha2

    v_rel = v - wind
    Fd = dragF(v_rel)

    T = m*g/np.cos(alpha)
    Tx = T*np.sin(alpha)*np.cos(theta)
    Ty = T*np.sin(alpha)*np.sin(theta)

    ax = (Tx - Fd[0])/m
    ay = (Ty - Fd[1])/m

    return [vx, vy, ax, ay]


def factors(u):
    theta1, alpha1, theta2, alpha2 = u


    sol = solve_ivp(lambda x,y: dynamics(x,y,u), [0, 10], [0,0,0,0], max_step=0.05)

    # compute energy
    energy = 0
    for i in range(len(sol.t)-1):
        x,y,vx,vy = sol.y[:,i]
        v_rel = np.array([vx,vy]) - wind
        alpha = alpha1 if sol.t[i] < 40 else alpha2
        P = Phover(alpha) + np.dot(dragF(v_rel), v_rel)
        dt = sol.t[i+1] - sol.t[i]
        energy += P*dt

    # penalty if not reaching target
    xf, yf = sol.y[0,-1], sol.y[1,-1]
    dist_error = np.linalg.norm([xf-target[0], yf-target[1]])

    return energy + 1e5*dist_error


initial_guess = [0.0, 0.2, 0.0, 0.2]

bounds = [
    (-np.pi, np.pi),   # theta1
    (0.05, 0.5),       # alpha1
    (-np.pi, np.pi),   # theta2
    (0.05, 0.5)        # alpha2
]

result = minimize(factors, initial_guess, bounds=bounds)
u = result.x
sol = solve_ivp(lambda x,y: dynamics(x,y,u), [0, 10], [0,0,0,0], max_step=0.05)

print("Optimal controls:", result.x)

print("Minimum energy:", result.fun)

plt.plot(sol.t,sol.y[0,:],label="x-position")
plt.plot(sol.t,sol.y[1,:],label="y-position")
plt.plot(sol.t,sol.y[2,:],label="x-velocity")
plt.plot(sol.t,sol.y[3,:],label="y-velocity")

plt.legend()
plt.show()

