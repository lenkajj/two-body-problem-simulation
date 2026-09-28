import pygame
import numpy as np
from scipy.integrate import solve_ivp


# ==================== PHYSICAL PARAMETERS ====================

m1 = 39993466555
m2 = 125634626556
G = 6.67428e-11

# Initial positions
x1, y1 = 5, 0
x2, y2 = -6, 0

# Initial velocities and angles
v1_0, theta1 = 0.376, np.pi / 3
v2_0, theta2 = 0.5, -np.pi

vx1_0 = v1_0 * abs(np.cos(theta1))
vy1_0 = v1_0 * np.sin(theta1)

vx2_0 = v2_0 * np.cos(theta2)
vy2_0 = v2_0 * np.sin(theta2)


# ==================== DIFFERENTIAL EQUATIONS ====================

def gravity(t, y):
    x1, y1, vx1, vy1, x2, y2, vx2, vy2 = y

    dx = x1 - x2
    dy = y1 - y2
    r = np.sqrt(dx**2 + dy**2)

    # Gravitational accelerations
    ax1 = G * m2 * (x2 - x1) / r**3
    ay1 = G * m2 * (y2 - y1) / r**3

    ax2 = G * m1 * (x1 - x2) / r**3
    ay2 = G * m1 * (y1 - y2) / r**3

    return [vx1, vy1, ax1, ay1, vx2, vy2, ax2, ay2]

initial_conditions = [x1, y1, vx1_0, vy1_0, x2, y2, vx2_0, vy2_0]


# ==================== PYGAME SETUP ====================

pygame.init()

width, height = 1000, 1000
screen = pygame.display.set_mode((width, height))

black = (0, 0, 0)
green = (0, 255, 0)
red = (255, 0, 0)

clock = pygame.time.Clock()
font = pygame.font.Font(None, 18)

running = True

# Planet trajectories
planet_path1 = []
planet_path2 = []

t = 0
dt = 0.1

# Scale: 1 unit of distance = 50 pixels
scale = 50


# ==================== MAIN LOOP ====================

while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Numerical solution of the equations of motion
    sol = solve_ivp(
        gravity,
        [t, t + dt],
        initial_conditions
    )

    t += dt
    new = sol.y[:, -1]

    # Current positions and velocities
    position_x1, position_y1, vx1, vy1, \
    position_x2, position_y2, vx2, vy2 = new

    initial_conditions = new

    # ==================== VELOCITIES ====================

    velocity1 = np.sqrt(vx1**2 + vy1**2)
    velocity2 = np.sqrt(vx2**2 + vy2**2)

    # ==================== DISTANCE BETWEEN BODIES ====================

    r = np.sqrt(
        (position_x2 - position_x1)**2 +
        (position_y2 - position_y1)**2
    )

    if r < 0.5:
        print("Bodies have collided.")
        running = False

    # ==================== CENTER OF MASS ====================

    # Center-of-mass velocity
    Vc_x = (m1 * vx1 + m2 * vx2) / (m1 + m2)
    Vc_y = (m1 * vy1 + m2 * vy2) / (m1 + m2)
    Vc = np.sqrt(Vc_x**2 + Vc_y**2)

    # Center-of-mass position
    # The camera is fixed to the center of mass
    view_x = (m1 * position_x1 + m2 * position_x2) / (m1 + m2)
    view_y = (m1 * position_y1 + m2 * position_y2) / (m1 + m2)

    # ==================== SCREEN COORDINATES ====================
    # The center-of-mass motion is removed from the visualization

    screen_x1 = int(
        width / 2 + (position_x1 - view_x) * scale
    )
    screen_y1 = int(
        height / 2 - (position_y1 - view_y) * scale
    )

    screen_x2 = int(
        width / 2 + (position_x2 - view_x) * scale
    )
    screen_y2 = int(
        height / 2 - (position_y2 - view_y) * scale
    )

    # Add the current positions to the trajectories
    planet_path1.append((screen_x1, screen_y1))
    planet_path2.append((screen_x2, screen_y2))

    # ==================== DRAWING ====================

    screen.fill((255, 255, 255))

    # Draw trajectories
    if len(planet_path1) > 1:
        pygame.draw.lines(
            screen, green, False, planet_path1, 1
        )

    if len(planet_path2) > 1:
        pygame.draw.lines(
            screen, red, False, planet_path2, 1
        )

    # Draw planets
    pygame.draw.circle(
        screen, green, (screen_x1, screen_y1), 10
    )

    pygame.draw.circle(
        screen, red, (screen_x2, screen_y2), 10
    )

    # ==================== TEXT INFORMATION ====================

    # Velocity of planet 1
    velocity1_text = font.render(
        f"v1 = {velocity1:.3f}", True, black
    )

    screen.blit(
        velocity1_text,
        (
            screen_x1 - velocity1_text.get_width() // 2,
            screen_y1 - 30
        )
    )

    # Velocity of planet 2
    velocity2_text = font.render(
        f"v2 = {velocity2:.3f}", True, black
    )

    screen.blit(
        velocity2_text,
        (
            screen_x2 - velocity2_text.get_width() // 2,
            screen_y2 - 30
        )
    )

    # Center-of-mass velocity
    Vc_text = font.render(
        f"Vc = {Vc:.3f}", True, black
    )

    screen.blit(
        Vc_text,
        (10, 10)
    )

    pygame.display.flip()
    clock.tick(60)


pygame.quit()
