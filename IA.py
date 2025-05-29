import numpy as np
import matplotlib
matplotlib.use('TkAgg') # Use TkAgg backend for interactive plots
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation

# --- Constants ---
e_charge = 1.602e-19  # Electron charge (C)
m_e = 9.109e-31       # Electron mass (kg)
E_exc_eV = 4.9        # Excitation energy of mercury (eV)
E_ret_eV = 1.0        # Retarding potential (eV) - not actively used in this simulation's current model
V_max = 15.0          # Maximum voltage for the slider (V)
tube_length = 0.1     # Length of the simulated tube (m)
electron_rate = 10    # Number of electrons spawned per frame
dt = 5e-8             # Time step for electron movement (s)
interval_ms = 20      # Animation update interval (ms)
collision_zones = np.linspace(0.02, 0.08, 5) # Visual zones where electrons might "collide"
star_life = 8         # Frames for a star to disappear

# --- I-V Curve Parameters ---
VALLE_WIDTH = 0.9   # Voltage range over which the current drops (V). Narrower = more abrupt drop.
VALLE_DEPTH = 0.9   # How deep the current drop is (0.9 = 90% loss in the center of the valley)

# --- Simulation Variables ---
x_e, y_e, vx_e, E_eV, color_e = [], [], [], [], [] # Electron properties
star_x, star_y, star_ttl = [], [], [] # Star (collision indicator) properties
voltages, I_counts = [], []           # Data for the I-V curve
prev_V = 0.0                          # Stores the previous voltage to detect changes

# --- Figure and Axes Setup ---
fig, (ax_tube, ax_iv) = plt.subplots(1, 2, figsize=(12, 5))
plt.subplots_adjust(bottom=0.25) # Make space for the slider

# --- Tube Plot (Left) ---
scatter_e = ax_tube.scatter([], [], s=12, c=[]) # Electron scatter plot
scatter_s = ax_tube.scatter([], [], s=70, marker='*', c='gold', alpha=0.9) # Star scatter plot
ax_tube.set_xlim(0, tube_length)
ax_tube.set_ylim(0, 1)
ax_tube.axis('off') # Hide axes for a cleaner tube visualization
ax_tube.set_title("Electrons in the Tube")

# --- I-V Curve Plot (Right) ---
scatter_iv = ax_iv.scatter([], [], c='k') # I-V curve data points
ax_iv.set_xlim(0, V_max)
ax_iv.set_ylim(0, electron_rate * 1.2) # Dynamic y-limit based on electron rate
ax_iv.set_xlabel("Voltage (V)")
ax_iv.set_ylabel("Current (e⁻/frame)")
ax_iv.set_title("I–V Curve")

# Add vertical dashed lines at excitation multiples
for n in range(1, int(V_max / E_exc_eV) + 1):
    vm = n * E_exc_eV
    if vm <= V_max:
        ax_iv.axvline(vm, color='red', linestyle='--', alpha=0.3)

# --- Voltage Slider ---
ax_sl = plt.axes([0.2, 0.06, 0.6, 0.03]) # Position for the slider
slider = Slider(ax_sl, 'Voltage (V)', 0, V_max, valinit=0)
current_V = 0.0 # Initial slider voltage

def on_slider(val):
    """Updates the current voltage when the slider is moved."""
    global current_V
    current_V = val
slider.on_changed(on_slider)

# --- Franck-Hertz Current Calculation ---
def franck_hertz_current(V):
    """
    Calculates the simulated current for the Franck-Hertz experiment.
    Incorporates drops at excitation voltages and subsequent higher peaks.
    """
    if V == 0:
        return 0

    n = int(V // E_exc_eV) # Number of excitation steps (multiples of E_exc_eV)
    V_local = V - n * E_exc_eV # Voltage within the current excitation step

    # Base current without any drops (linear increase with total voltage)
    I_base_no_drops = electron_rate * (V / E_exc_eV)

    # Calculate penalty only near the excitation voltage (valley region)
    penalty = 0.0
    if V_local < VALLE_WIDTH:
        # Gaussian-like drop centered within the VALLE_WIDTH
        # The (VALLE_WIDTH/4) in the denominator controls the steepness of the drop
        penalty = VALLE_DEPTH * np.exp(-((V_local - VALLE_WIDTH/2)/(VALLE_WIDTH/4))**2)

    # Apply the penalty to the base current
    I_current = I_base_no_drops * (1 - penalty)

    # Ensure current doesn't go below a certain threshold to avoid zero current
    # This also gives a slight increasing baseline after each drop
    I_current = max(I_current, electron_rate * 0.1 * (n + 0.5)) # Added 0.5 to 'n' for a non-zero minimum at n=0

    # Make the current rise to a higher peak after each excitation step
    # The '0.15' factor can be adjusted to control the peak increase
    I_current *= (1 + n * 0.15)

    return I_current

# --- Animation Function ---
frame = 0
def animate(_):
    """
    Updates the electron simulation and I-V curve data for each animation frame.
    """
    global frame, x_e, y_e, vx_e, E_eV, color_e, star_x, star_y, star_ttl
    global voltages, I_counts, prev_V
    frame += 1

    # --- Electron Simulation (Tube) ---
    # Clear electrons if voltage changed significantly, for visual realism
    # Or if too many electrons accumulate (performance safeguard)
    if abs(current_V - prev_V) > 0.5 or len(x_e) > 500:
        x_e.clear(); y_e.clear(); vx_e.clear(); E_eV.clear(); color_e.clear()

    # Spawn new electrons
    for _ in range(electron_rate):
        x_e.append(0.0) # Start at the beginning of the tube
        y_e.append(np.random.rand()) # Random Y position
        # Initial energy based on the local voltage within the current excitation step
        n = int(current_V // E_exc_eV)
        V_base = n * E_exc_eV
        V_local_initial = current_V - V_base
        E_eV.append(V_local_initial) # Store energy in eV
        # Calculate initial velocity from local initial energy
        v0 = np.sqrt(2 * V_local_initial * e_charge / m_e) if V_local_initial > 0 else 0
        vx_e.append(v0)
        color_e.append('blue') # Default color for new electrons

    # Convert lists to NumPy arrays for efficient calculations
    x = np.array(x_e)
    y = np.array(y_e)
    vx = np.array(vx_e)
    E = np.array(E_eV, dtype=float)
    col = np.array(color_e, dtype=object)

    # Move electrons
    # In a simplified model, velocity is constant within a step until collision/excitation
    x += vx * dt

    # Collision visualization: electrons near excitation points change color and leave a star
    for z in collision_zones:
        # Mask for electrons that have passed a collision zone AND have significant energy
        # and haven't already "collided" (still blue)
        mask_hit = (x >= z) & (E > 0.1) & (col == 'blue')
        col[mask_hit] = 'orange' # Change color to indicate energy loss
        # Add a star at the collision point for visual effect
        for i in np.where(mask_hit)[0]:
            star_x.append(x[i])
            star_y.append(y[i])
            star_ttl.append(star_life) # Set star's time-to-live

    # Update and remove old stars
    if star_ttl:
        ttl = np.array(star_ttl) - 1
        keep = ttl > 0 # Keep stars with time-to-live > 0
        star_x = list(np.array(star_x)[keep])
        star_y = list(np.array(star_y)[keep])
        star_ttl = list(ttl[keep])

    # Remove electrons that have left the tube
    keep_e = x < tube_length
    x_e = x[keep_e].tolist()
    y_e = y[keep_e].tolist()
    vx_e = vx[keep_e].tolist()
    E_eV = E[keep_e].tolist()
    color_e = col[keep_e].tolist()

    # Update scatter plots
    scatter_e.set_offsets(np.column_stack([x_e, y_e]))
    scatter_e.set_color(color_e)
    # Ensure scatter_s handles empty arrays gracefully
    scatter_s.set_offsets(np.column_stack([star_x, star_y]) if star_x else np.empty((0, 2)))

    # --- I-V Curve Update ---
    I = franck_hertz_current(current_V) # Calculate current using the new function

    # Only add a new point if the voltage has changed or it's the very first point
    if current_V != prev_V or not voltages:
        voltages.append(current_V)
        I_counts.append(I)
        # Update the I-V curve scatter plot
        scatter_iv.set_offsets(np.column_stack([voltages, I_counts]))
        prev_V = current_V # Update previous voltage

    # Return all artists that were modified
    return scatter_e, scatter_s, scatter_iv

# --- Run Animation ---a
ani = FuncAnimation(fig, animate, interval=interval_ms, blit=False) # blit=False for TkAgg
plt.show()
#Hola