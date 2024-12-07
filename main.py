import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Arrow:
    def __init__(self, color, line_style='-'):
        self.arrow_template_radii = np.array([0.25, 0, 0.25])
        self.arrow_template_angles = np.array([np.pi * 5 / 6, 0, np.pi * 7 / 6])
        self.body, = region.plot(np.array([]), np.array([]), color=color, linestyle=line_style)
        self.head, = region.plot(np.array([]), np.array([]), color=color, linestyle=line_style)

    def update(self, x1, y1, x2, y2):
        self.body.set_data(np.array([x1, x2]), np.array([y1, y2]))
        angle = np.arctan2(y2 - y1, x2 - x1)
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        head_x, head_y = polar_coordinates_to_cartesian(
            self.arrow_template_radii * length,
            self.arrow_template_angles + np.full(3, angle)
        )
        self.head.set_data(np.full(3, x2) + head_x, np.full(3, y2) + head_y)

    def return_plot(self):
        return self.body, self.head


def polar_coordinates_to_cartesian(r, fi):
    """Converts polar coordinates to cartesian coordinates"""
    x = r * np.cos(fi)
    y = r * np.sin(fi)
    return x, y


def polar_velocities_to_cartesian(v_r, v_fi, r, fi):
    """Converts polar velocities to cartesian velocities"""
    v_x = v_r * np.cos(fi) - v_fi * r * np.sin(fi)
    v_y = v_r * np.sin(fi) + v_fi * r * np.cos(fi)
    return v_x, v_y


def polar_accelerations_to_cartesian(a_r, a_fi, r, fi):
    """Converts polar accelerations to cartesian accelerations"""
    a_x = a_r * np.cos(fi) - a_fi * r * np.sin(fi)
    a_y = a_r * np.sin(fi) + a_fi * r * np.cos(fi)
    return a_x, a_y


def calculate_values(expression: sp.Expr, time_points: np.ndarray) -> np.ndarray:
    """Calculates the values of the given expression
    at the given time points"""
    values = sp.lambdify(t, expression, 'numpy')(time_points)
    if isinstance(values, int | float):
        values = np.full(time_points.size, values)
    return values


def calculate_tangential_acceleration(acceleration_x, acceleration_y, velocity_x, velocity_y):
    """Calculates tangential acceleration"""
    velocity = np.sqrt(velocity_x ** 2 + velocity_y ** 2)
    tangential = (acceleration_x * velocity_x + acceleration_y * velocity_y) / velocity
    tangential_x = tangential * (velocity_x / velocity)
    tangential_y = tangential * (velocity_y / velocity)
    return tangential_x, tangential_y, tangential


def calculate_centripetal_acceleration(tangential, acceleration_x, acceleration_y, velocity_x, velocity_y):
    """Calculates centripetal acceleration"""
    orientation = velocity_x * acceleration_y - velocity_y * acceleration_x
    if orientation == 0:
        return 0, 0
    centripetal = np.sqrt(acceleration_x ** 2 + acceleration_y ** 2 - tangential ** 2)
    velocity = np.sqrt(velocity_x ** 2 + velocity_y ** 2)
    centripetal_x = -centripetal * (velocity_y / velocity)
    centripetal_y = centripetal * (velocity_x / velocity)
    if orientation < 0:
        centripetal_x *= -1
        centripetal_y *= -1
    return centripetal_x, centripetal_y


# Conditions
max_time = 30

t = sp.Symbol('t')  # time
radius: sp.Expr = sp.cos(t) + 1
angle: sp.Expr = t * 5 / 4

velocity_radius = sp.diff(radius, t)
velocity_angle = sp.diff(angle, t)

acceleration_radius = sp.diff(velocity_radius, t)
acceleration_angle = sp.diff(velocity_angle, t)

# Calculations
time_points = np.linspace(0, max_time, 1000)

radius_values = calculate_values(radius, time_points)
angle_values = calculate_values(angle, time_points)

radius_velocity_values = calculate_values(velocity_radius, time_points)
angle_velocity_values = calculate_values(velocity_angle, time_points)

radius_acceleration_values = calculate_values(acceleration_radius, time_points)
angle_acceleration_values = calculate_values(acceleration_angle, time_points)

# Rendering
window = plt.figure()
region = window.add_subplot(1, 1, 1)
region.set_title("Вариант 20, Сергеев Владимир")
region.axis('image')
axes_limit = radius_values.max() * 2
region.set_xlim(-axes_limit, axes_limit)
region.set_ylim(-axes_limit, axes_limit)

curve, = region.plot(np.array([]), np.array([]))
body, = region.plot(np.array([]), np.array([]), marker='o')
velocity_arrow = Arrow('red')
acceleration_arrow = Arrow('purple', line_style=':')
tangential_acceleration_arrow = Arrow('green', line_style='--')
centripetal_acceleration_arrow = Arrow('blue', line_style='--')

fps = 60


def animate(i):
    if i == time_points.size:
        ani.event_source.stop()
        print("Симуляция завершена")
        return curve, body, *velocity_arrow.return_plot(), *acceleration_arrow.return_plot(), \
            *tangential_acceleration_arrow.return_plot(), *centripetal_acceleration_arrow.return_plot()

    x, y = polar_coordinates_to_cartesian(radius_values[:i + 1], angle_values[:i + 1])
    curve.set_data(x, y)
    body.set_data(np.array([x[-1]]), np.array([y[-1]]))

    velocity_x, velocity_y = polar_velocities_to_cartesian(radius_velocity_values[i], angle_velocity_values[i],
                                                           radius_values[i], angle_values[i])
    velocity_arrow.update(x[-1], y[-1], x[-1] + velocity_x, y[-1] + velocity_y)

    acceleration_x, acceleration_y = polar_accelerations_to_cartesian(
        radius_acceleration_values[i], angle_acceleration_values[i],
        radius_values[i], angle_values[i])
    acceleration_arrow.update(x[-1], y[-1], x[-1] + acceleration_x, y[-1] + acceleration_y)

    tangential_x, tangential_y, tangential = calculate_tangential_acceleration(acceleration_x, acceleration_y,
                                                                               velocity_x, velocity_y)
    tangential_acceleration_arrow.update(x[-1], y[-1], x[-1] + tangential_x, y[-1] + tangential_y)

    centripetal_x, centripetal_y = calculate_centripetal_acceleration(tangential, acceleration_x, acceleration_y,
                                                                      velocity_x, velocity_y)
    centripetal_acceleration_arrow.update(x[-1], y[-1], x[-1] + centripetal_x, y[-1] + centripetal_y)

    return curve, body, *velocity_arrow.return_plot(), *acceleration_arrow.return_plot(), \
        *tangential_acceleration_arrow.return_plot(), *centripetal_acceleration_arrow.return_plot()


ani = animation.FuncAnimation(window, animate, frames=time_points.size + 1, interval=round(1000 / fps, 0), blit=True)
plt.show()
