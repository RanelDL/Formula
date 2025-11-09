import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from simple_pid import PID
import sys

class Vehicle:
    """
    A simple vehicle simulation model.
    The vehicle is modeled as a point mass with aerodynamic drag.
    """
    def __init__(self, mass=250, drag_coeff=0.7, max_force=4000):
        self.mass = mass              # kg
        self.drag_coeff = drag_coeff  # unitless
        self.C_rr = 0.015             # rolling resistance of tires
        self.max_force = max_force    # Newtons (max engine force)
        self.max_power_watts = 80000  # 80kW 

        self.velocity = 0             # m/s
        self.position = 0             # m
        self.base_speed_ms = 22.0

    def get_propulsion_force(self, throttle):

        # Propulsive force is limited by the throttle
        available_power = self.max_power_watts * throttle

        if self.velocity < self.base_speed_ms:
            # --- Regime 1: Constant Torque (Force) ---
            # At low speed, we use the pre-calculated maximum force, scaled by throttle.
            # We also need to make sure power doesn't exceed the available power limit.
            force = self.max_propulsive_force * throttle
            
            # The actual power being used is Force * Velocity.
            # If this exceeds the power limit from the throttle, we must cap the force.
            if self.velocity > 0: # Avoid division by zero
                force = min(force, available_power / self.velocity)
            
        else:
            # --- Regime 2: Constant Power ---
            # At high speed, force is limited by the available power.
            if self.velocity > 0:
                force = available_power / self.velocity
            else:
                force = self.max_propulsive_force * throttle # Should not happen if v > base_speed
        
        return force

    def get_resistive_force(self, current_speed):
        # Aerodynamic drag force is proportional to the square of velocity
        drag_force = self.drag_coeff * current_speed**2

        friction_force = self.C_rr * self.mass * 9.81
        return drag_force + friction_force
    
    def update(self, throttle, dt):
        """
        Update the vehicle's state after a time step dt.
        
        Args:
            throttle (float): The throttle input, from 0.0 to 1.0.
            dt (float): The time step in seconds.
        """
        # Clamp throttle between 0 and 1
        throttle = max(0.0, min(1.0, throttle))
        
        # Calculate forces
        # engine_force = self.get_propulsion_force(throttle) #throttle * self.max_force
        engine_force = throttle * self.max_force
        
        resistive_force = self.get_resistive_force(self.velocity)
        
        # Total force is engine force minus drag
        total_force = engine_force - resistive_force 
        
        # Calculate acceleration (F=ma)
        acceleration = total_force / self.mass
        
        # Update velocity and position
        self.velocity += acceleration * dt
        self.position += self.velocity * dt
        
        # Don't let the car go backwards if drag is stronger than a 0-throttle engine
        if self.velocity < 0:
            self.velocity = 0

# --- Simulation Setup ---
TARGET_SPEED = 25.0  # Target speed in m/s (approx 90 km/h or 56 mph)
SIMULATION_TIME = 30 # seconds
DT = 0.05             # Time step in seconds (5ms)

# --- PID Controller Setup ---
# These Kp, Ki, Kd values are the "gains" you need to tune.
Kp = 0.5   # Proportional gain: Reacts to current error.
Ki = 0.1    # Integral gain: Corrects for past, accumulated error.
Kd = 0.2     # Derivative gain: Predicts future error, dampens oscillations.
    
pid = PID(Kp, Ki, Kd, setpoint=TARGET_SPEED, sample_time=DT)
# Set the output limits of the PID controller (0% to 100% throttle)
pid.output_limits = (0, 1.0)

# --- Vehicle Setup ---
car = Vehicle()

# --- Data for plotting ---
time_points = []
speed_points = []
throttle_points = []
setpoint_points = []

# --- Set up the plot ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

def animate(frame):
    # 1. Get the current speed of the car
    current_speed = car.velocity
        
    # 2. Calculate the PID output (the new throttle setting)
    throttle = pid(current_speed)

    # 2.5: Calculate feed-forward term for throttle smoothing
    feed_forward_throttle = car.get_resistive_force(TARGET_SPEED)/car.max_force

    # throttle += feed_forward_throttle #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
    # 3. Update the car's state using the new throttle
    car.update(throttle, DT)
        
    # 4. Store data for plotting
    current_time = frame * DT
    time_points.append(current_time)
    speed_points.append(current_speed)
    throttle_points.append(throttle)
    setpoint_points.append(TARGET_SPEED)
    
    # 5. Clear the axes and redraw the plots
    ax1.clear()
    ax2.clear()
    
    # Plot Speed vs. Time
    ax1.plot(time_points, speed_points, label='Vehicle Speed')
    ax1.plot(time_points, setpoint_points, 'r--', label='Target Speed')
    ax1.set_ylabel('Speed (m/s)')
    ax1.set_title('Cruise Control PID Simulation')
    ax1.legend()
    ax1.grid(True)

    # Plot Throttle vs. Time
    ax2.plot(time_points, [t * 100 for t in throttle_points], label='Throttle', color='green')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Throttle (%)')
    ax2.legend()
    ax2.grid(True)

# --- Main execution ---
if __name__ == "__main__":
    print("Running cruise control simulation with live plotting...")
    
    # Create the animation
    ani = FuncAnimation(fig, animate, frames=int(SIMULATION_TIME / DT), interval=DT*1000, repeat=False)
    
    plt.show()

    print("Simulation complete.")