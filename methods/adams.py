def adams_method(
    f, initial_x, initial_y, target_x, num_steps, tolerance, exact_solution
):
    print("Adams method:")

    step_size = (target_x - initial_x) / num_steps
    x_values = [initial_x]
    y_values = [initial_y]
    derivative_values = []

    initial_condition_x = initial_x
    initial_condition_y = initial_y

    # Calculate first 4 points using Runge-Kutta
    for i in range(min(3, num_steps)):
        current_x = x_values[-1]
        current_y = y_values[-1]

        k1 = step_size * f(current_x, current_y)
        k2 = step_size * f(current_x + step_size / 2, current_y + k1 / 2)
        k3 = step_size * f(current_x + step_size / 2, current_y + k2 / 2)
        k4 = step_size * f(current_x + step_size, current_y + k3)

        next_y = current_y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        next_x = current_x + step_size

        x_values.append(next_x)
        y_values.append(next_y)

    if num_steps <= 3:
        print(
            "\nAdams method requires at least 4 points. Displaying Runge-Kutta results."
        )
        print_results(
            x_values,
            y_values,
            exact_solution,
            initial_condition_x,
            initial_condition_y,
            tolerance,
        )
        return x_values, y_values

    # Store derivatives for initial points
    for i in range(4):
        derivative_values.append(f(x_values[i], y_values[i]))

    # Adams-Bashforth-Moulton predictor-corrector
    for i in range(3, num_steps):
        # Predictor (Adams-Bashforth)
        y_predicted = y_values[i] + step_size / 24 * (
            55 * derivative_values[i]
            - 59 * derivative_values[i - 1]
            + 37 * derivative_values[i - 2]
            - 9 * derivative_values[i - 3]
        )
        next_x = x_values[i] + step_size

        f_predicted = f(next_x, y_predicted)

        # Corrector (Adams-Moulton)
        y_corrected = y_values[i] + step_size / 24 * (
            9 * f_predicted
            + 19 * derivative_values[i]
            - 5 * derivative_values[i - 1]
            + derivative_values[i - 2]
        )

        x_values.append(next_x)
        y_values.append(y_corrected)
        derivative_values.append(f(next_x, y_corrected))

    print("\nAdams predictor-corrector results:")
    print("x\t\tApproximate y\t\tExact y\t\t\tAbsolute Error")
    for i in range(len(x_values)):
        exact = exact_solution(x_values[i], initial_condition_x, initial_condition_y)
        error = abs(y_values[i] - exact)
        print(f"{x_values[i]:.6f}\t{y_values[i]:.12f}\t{exact:.12f}\t{error:.2e}")

    final_error = abs(
        y_values[-1]
        - exact_solution(target_x, initial_condition_x, initial_condition_y)
    )
    if final_error < tolerance:
        print(
            f"\nAccuracy achieved: |approx_y - exact_y| = {final_error:.2e} < ε = {tolerance}"
        )
    else:
        print(
            f"\nAccuracy NOT achieved: |approx_y - exact_y| = {final_error:.2e} >= ε = {tolerance}"
        )

    return x_values, y_values


def print_results(x_values, y_values, exact_solution, initial_x, initial_y, tolerance):
    print("x\t\tApproximate y\t\tExact y\t\t\tAbsolute Error")
    for i in range(len(x_values)):
        exact = exact_solution(x_values[i], initial_x, initial_y)
        error = abs(y_values[i] - exact)
        print(f"{x_values[i]:.6f}\t{y_values[i]:.12f}\t{exact:.12f}\t{error:.2e}")

    final_error = abs(y_values[-1] - exact_solution(x_values[-1], initial_x, initial_y))
    if final_error < tolerance:
        print(
            f"\nAccuracy achieved: |approx_y - exact_y| = {final_error:.2e} < ε = {tolerance}"
        )
    else:
        print(
            f"\nAccuracy NOT achieved: |approx_y - exact_y| = {final_error:.2e} >= ε = {tolerance}"
        )
