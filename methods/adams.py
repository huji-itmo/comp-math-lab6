def adams_method(f, x0, y0, xn, n, epsilon, exact_solution):
    print("Adams-Bashforth-Moulton predictor-corrector method:")

    h = (xn - x0) / n
    x_points = [x0]
    y_points = [y0]
    f_values = []

    initial_x = x0
    initial_y = y0

    # Initial steps using Runge-Kutta 4th order
    for i in range(min(3, n)):
        xi = x_points[-1]
        yi = y_points[-1]

        k1 = h * f(xi, yi)
        k2 = h * f(xi + h / 2, yi + k1 / 2)
        k3 = h * f(xi + h / 2, yi + k2 / 2)
        k4 = h * f(xi + h, yi + k3)

        next_y = yi + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        next_x = xi + h

        x_points.append(next_x)
        y_points.append(next_y)

    if n <= 3:
        print("\nAdams method requires ≥4 points. Showing Runge-Kutta results.")
        display_results(
            x_points, y_points, exact_solution, initial_x, initial_y, epsilon
        )
        return x_points, y_points

    # Store function values for Adams
    for i in range(4):
        f_values.append(f(x_points[i], y_points[i]))

    # Adams predictor-corrector iterations
    for i in range(3, n):
        # Predictor (Adams-Bashforth)
        y_pred = y_points[i] + h / 24 * (
            55 * f_values[i]
            - 59 * f_values[i - 1]
            + 37 * f_values[i - 2]
            - 9 * f_values[i - 3]
        )
        x_next = x_points[i] + h

        # Corrector (Adams-Moulton)
        f_pred = f(x_next, y_pred)
        y_corrected = y_points[i] + h / 24 * (
            9 * f_pred + 19 * f_values[i] - 5 * f_values[i - 1] + f_values[i - 2]
        )

        x_points.append(x_next)
        y_points.append(y_corrected)
        f_values.append(f(x_next, y_corrected))

    print("\nAdams method results (predictor-corrector):")
    print_results(x_points, y_points, exact_solution, initial_x, initial_y, epsilon)

    return x_points, y_points


def display_results(x_points, y_points, exact_solution, x0, y0, epsilon):
    print("x\t\tApproximate y\t\tExact solution\t\tAbsolute error")
    for i in range(len(x_points)):
        exact = exact_solution(x_points[i], x0, y0)
        error = abs(y_points[i] - exact)
        print(f"{x_points[i]:.6f}\t{y_points[i]:.12f}\t{exact:.12f}\t{error:.2e}")

    final_error = abs(y_points[-1] - exact_solution(x_points[-1], x0, y0))
    if final_error < epsilon:
        print(f"\nAccuracy achieved: |Δy| = {final_error:.2e} < ε = {epsilon}")
    else:
        print(f"\nAccuracy NOT achieved: |Δy| = {final_error:.2e} ≥ ε = {epsilon}")


def print_results(x_points, y_points, exact_solution, x0, y0, epsilon):
    print("x\t\tApproximate y\t\tExact solution\t\tAbsolute error")
    for i in range(len(x_points)):
        exact = exact_solution(x_points[i], x0, y0)
        error = abs(y_points[i] - exact)
        print(f"{x_points[i]:.6f}\t{y_points[i]:.12f}\t{exact:.12f}\t{error:.2e}")

    final_error = abs(y_points[-1] - exact_solution(x_points[-1], x0, y0))
    if final_error < epsilon:
        print(f"\nAccuracy achieved: |Δy| = {final_error:.2e} < ε = {epsilon}")
    else:
        print(f"\nAccuracy NOT achieved: |Δy| = {final_error:.2e} ≥ ε = {epsilon}")
