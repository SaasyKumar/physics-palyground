import taichi as ti
import numpy as np
import math

try:
    ti.init(arch=ti.gpu)
except Exception:
    ti.init(arch=ti.cpu)
    print("Running on CPU, expect lower FPS. Lower the particle count if it crawls.")

# Configuration Constants
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 800
NUM_PARTICLES = 65536  # Power of 2 for GPU optimization

# Physics Constants
G = 0.1  # Gravitational constant
M_BH = 1.5  # Mass of the black hole
EPS = 1e-3  # Zero division protection epsilon
R_EVENT_HORIZON = 0.025  # Event horizon radius (absorption threshold)

# Simulation fields
pos = ti.Vector.field(2, dtype=ti.f32, shape=NUM_PARTICLES)
vel = ti.Vector.field(2, dtype=ti.f32, shape=NUM_PARTICLES)
color = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)

# Central Black Hole state
bh_pos = ti.Vector([0.5, 0.5])

# Time control
dt = ti.field(dtype=ti.f32, shape=())
paused = ti.field(dtype=ti.i32, shape=())

@ti.kernel
def initialize_particles():
    for i in range(NUM_PARTICLES):
        # We mix orbital particles (70%) and infalling/random particles (30%)
        is_orbit = ti.random() < 0.7
        
        if is_orbit:
            # Circular Keplerian orbit initialization
            r = ti.random() * 0.35 + 0.05
            theta = ti.random() * 2.0 * math.pi
            
            p = ti.Vector([
                0.5 + r * ti.cos(theta),
                0.5 + r * ti.sin(theta)
            ])
            pos[i] = p
            
            # Keplerian velocity: v = sqrt(G * M / r)
            v_mag = ti.sqrt(G * M_BH / (r + EPS))
            # Direction is perpendicular to radius vector
            vel[i] = ti.Vector([-ti.sin(theta), ti.cos(theta)]) * v_mag * (1.0 + 0.1 * ti.random())
        else:
            # Streaming particles from outer bounds
            theta = ti.random() * 2.0 * math.pi
            r = 0.5  # start at the boundary
            pos[i] = ti.Vector([
                0.5 + r * ti.cos(theta),
                0.5 + r * ti.sin(theta)
            ])
            # Directing them roughly towards the center or slinging past it
            target_offset = ti.Vector([ti.random() - 0.5, ti.random() - 0.5]) * 0.15
            dir_vec = (ti.Vector([0.5, 0.5]) + target_offset - pos[i]).normalized()
            v_mag = ti.random() * 0.3 + 0.1
            vel[i] = dir_vec * v_mag

@ti.func
def respawn_particle(i: ti.i32):
    # Respawn particle on absorption or escape
    is_orbit = ti.random() < 0.8
    if is_orbit:
        r = ti.random() * 0.3 + 0.15
        theta = ti.random() * 2.0 * math.pi
        pos[i] = ti.Vector([
            0.5 + r * ti.cos(theta),
            0.5 + r * ti.sin(theta)
        ])
        v_mag = ti.sqrt(G * M_BH / (r + EPS))
        vel[i] = ti.Vector([-ti.sin(theta), ti.cos(theta)]) * v_mag * (1.0 + 0.05 * ti.random())
    else:
        theta = ti.random() * 2.0 * math.pi
        r = 0.5
        pos[i] = ti.Vector([
            0.5 + r * ti.cos(theta),
            0.5 + r * ti.sin(theta)
        ])
        target_offset = ti.Vector([ti.random() - 0.5, ti.random() - 0.5]) * 0.1
        dir_vec = (ti.Vector([0.5, 0.5]) + target_offset - pos[i]).normalized()
        v_mag = ti.random() * 0.2 + 0.1
        vel[i] = dir_vec * v_mag

@ti.kernel
def update_particles(mouse_pos: ti.types.vector(2, ti.f32), mouse_state: ti.i32):
    current_dt = dt[None]
    
    for i in range(NUM_PARTICLES):
        p = pos[i]
        v = vel[i]
        
        # 1. Newtonian Gravitational Force from central Black Hole
        to_bh = bh_pos - p
        dist_bh_sq = to_bh.norm_sqr() + EPS
        dist_bh = ti.sqrt(dist_bh_sq)
        
        # Acceleration: a = G * M_BH / dist^2
        acc = to_bh * (G * M_BH / (dist_bh_sq * dist_bh))
        
        # 2. Mouse Interaction (Gravity Pull or Push)
        if mouse_state > 0:
            to_mouse = mouse_pos - p
            dist_mouse_sq = to_mouse.norm_sqr() + EPS
            dist_mouse = ti.sqrt(dist_mouse_sq)
            if dist_mouse < 0.4:
                # Pull if left click (state 1), push if right click (state 2)
                strength = 0.15 if mouse_state == 1 else -0.15
                acc += to_mouse * (strength / (dist_mouse_sq * dist_mouse))
        
        # 3. Semi-implicit Euler integration
        v += acc * current_dt
        p += v * current_dt
        
        # Write back
        vel[i] = v
        pos[i] = p
        
        # Check boundary & absorption conditions
        dist_to_center = (p - bh_pos).norm()
        if dist_to_center < R_EVENT_HORIZON or dist_to_center > 0.8:
            respawn_particle(i)
            
        # 4. Physical color calculation based on speed
        speed = v.norm()
        # Map speed to color spectrum (Blue/Violet -> Pink/Magenta -> Orange/White)
        # Slow = deep blue, fast = white hot
        t = ti.min(speed / 2.5, 1.0)
        
        r_col = ti.pow(t, 2.0) * 1.5
        g_col = ti.pow(t, 4.0) * 1.2
        b_col = ti.pow(1.0 - t, 0.5) * 0.8 + ti.pow(t, 3.0) * 1.2
        
        # Enhance brightness for very close/fast particles
        brightness = 0.5 + 0.5 * t
        color[i] = ti.Vector([r_col, g_col, b_col]) * brightness

# Print controls
print("=" * 60)
print(" Newtonian Black Hole Simulation (No Accretion Disk)")
print(" Controls:")
print("   [SPACE]        - Pause / Resume simulation")
print("   [ArrowUp]/[+]  - Speed up simulation time")
print("   [ArrowDown]/[-] - Slow down simulation time")
print("   [R]            - Reset particle positions")
print("   [Left Click]   - Gravitational pull towards mouse cursor")
print("   [Right Click]  - Gravitational repulsion from mouse cursor")
print("=" * 60)

# Init fields
dt[None] = 0.05
paused[None] = 0
initialize_particles()

# Window Setup
window = ti.ui.Window("Newtonian Black Hole", (WINDOW_WIDTH, WINDOW_HEIGHT))
canvas = window.get_canvas()

while window.running:
    # Handle Keyboard & Mouse events
    if window.get_event(ti.ui.PRESS):
        event = window.event
        if event.key == ti.ui.SPACE:
            paused[None] = 1 - paused[None]
        elif event.key == 'r' or event.key == 'R':
            initialize_particles()
        elif event.key == ti.ui.UP or event.key == '+':
            dt[None] = min(dt[None] * 1.2, 0.2)
            print(f"Time speed: {dt[None]:.4f}")
        elif event.key == ti.ui.DOWN or event.key == '-':
            dt[None] = max(dt[None] / 1.2, 0.002)
            print(f"Time speed: {dt[None]:.4f}")

    # Mouse state detection
    mouse_pos = window.get_cursor_pos()
    mouse_state = 0
    if window.is_pressed(ti.ui.LMB):
        mouse_state = 1
    elif window.is_pressed(ti.ui.RMB):
        mouse_state = 2

    # Step simulation if not paused
    if not paused[None]:
        update_particles(ti.Vector([mouse_pos[0], mouse_pos[1]]), mouse_state)

    # Render
    canvas.set_background_color((0.005, 0.005, 0.008))
    
    # Draw particles (use a very small radius for glow field effect)
    canvas.circles(pos, radius=0.0012, per_vertex_color=color)
    
    # Draw the central Black Hole mass (event horizon boundary)
    # We draw a series of concentric circles for a nice glowing/fading horizon effect
    # The event horizon itself is pitch black
    bh_color_field = ti.Vector.field(3, dtype=ti.f32, shape=1)
    bh_pos_field = ti.Vector.field(2, dtype=ti.f32, shape=1)
    bh_pos_field[0] = bh_pos
    
    # Outer horizon glow
    bh_color_field[0] = ti.Vector([0.8, 0.2, 0.5])
    canvas.circles(bh_pos_field, radius=R_EVENT_HORIZON * 1.2, per_vertex_color=bh_color_field)
    
    # Inner horizon glow
    bh_color_field[0] = ti.Vector([0.2, 0.4, 0.8])
    canvas.circles(bh_pos_field, radius=R_EVENT_HORIZON * 1.05, per_vertex_color=bh_color_field)
    
    # Actual Event Horizon (black mass)
    bh_color_field[0] = ti.Vector([0.0, 0.0, 0.0])
    canvas.circles(bh_pos_field, radius=R_EVENT_HORIZON, per_vertex_color=bh_color_field)

    window.show()
