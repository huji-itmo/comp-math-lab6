import math

from methods.adams import adams_method
from methods.euler import euler_method
from methods.runge_kutta import runge_kutt
from plot import plot_graphic


def select_ode():
    print("Select an Ordinary Differential Equation (ODE):")
    print("1. y' = x + y")
    print("2. y' = sin(x) - y")
    print("3. y' = y / x")

    while True:
        try:
            selection = int(input("> Select ODE [1/2/3]: "))
            if selection == 1:
                ode_func = lambda x, y: x + y
                exact_solution = (
                    lambda x, ref_x, ref_y: math.exp(x - ref_x) * (ref_y + ref_x + 1)
                    - x
                    - 1
                )
                return selection, ode_func, exact_solution
            elif selection == 2:
                ode_func = lambda x, y: math.sin(x) - y
                exact_solution = (
                    lambda x, ref_x, ref_y: (
                        ref_y - 0.5 * math.sin(ref_x) + 0.5 * math.cos(ref_x)
                    )
                    * math.exp(ref_x - x)
                    + 0.5 * math.sin(x)
                    - 0.5 * math.cos(x)
                )
                return selection, ode_func, exact_solution
            elif selection == 3:
                ode_func = lambda x, y: y / x
                exact_solution = lambda x, ref_x, ref_y: ref_y * math.exp(
                    math.log(x) - math.log(ref_x)
                )
                return selection, ode_func, exact_solution
            else:
                print("Invalid input! Please try again")
        except ValueError:
            print("Invalid input! Please enter a number 1-3")


def read_parameters(ode_selection):
    while True:
        try:
            start_x = float(input("Enter interval start point x0: "))
            end_x = float(input("Enter interval end point xn: "))
            # Validate non-zero interval length
            if start_x == end_x:
                print(
                    "Error: Interval length is zero! Start and end points must differ."
                )
                continue  # Restart the loop to get new inputs

            step_count = int(input("Enter initial number of steps: "))
            initial_y = float(input("Enter initial condition y0: "))
            tolerance = float(input("Enter solution tolerance (epsilon): "))

            # Validate other parameters
            if step_count <= 0:
                print("Step count must be positive!")
                continue
            if tolerance <= 0:
                print("Tolerance must be positive!")
                continue
            if start_x == 0 and ode_selection == 3:
                print("Error: x=0 is not allowed for ODE 3 (division by zero)")
                continue

            return start_x, end_x, step_count, initial_y, tolerance
        except ValueError:
            print("Invalid input! Please enter numerical values")
        except ZeroDivisionError:
            print("Error: Initial x cannot be zero for ODE 3")


def main():
    # Get ODE selection and functions
    ode_selection, f, exact_solution = select_ode()

    # Read parameters with ODE selection for validation
    start_x, end_x, step_count, initial_y, tolerance = read_parameters(ode_selection)

    # Call solvers (these need to be defined)
    x_values_runge, y_values_runge = runge_kutt(
        f, start_x, initial_y, end_x, step_count, tolerance, exact_solution
    )
    x_values_euler, y_values_euler = euler_method(
        f, start_x, initial_y, end_x, step_count, tolerance, exact_solution
    )
    x_values_adams, y_values_adams = adams_method(
        f, start_x, initial_y, end_x, step_count, tolerance, exact_solution
    )

    plot_graphic(
        x_values_runge, y_values_runge, exact_solution, start_x, initial_y, "runge_kutt"
    )
    plot_graphic(
        x_values_euler, y_values_euler, exact_solution, start_x, initial_y, "euler"
    )
    plot_graphic(
        x_values_adams, y_values_adams, exact_solution, start_x, initial_y, "adams"
    )


if __name__ == "__main__":
    main()
