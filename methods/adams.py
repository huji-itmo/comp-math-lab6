def adams_predictor(current_y, h, f_current, f_prev1, f_prev2, f_prev3):
    return current_y + h / 24 * (
        55 * f_current - 59 * f_prev1 + 37 * f_prev2 - 9 * f_prev3
    )


def adams_corrector_step(current_y, h, f_pred, f_current, f_prev1, f_prev2):
    return current_y + h / 24 * (9 * f_pred + 19 * f_current - 5 * f_prev1 + f_prev2)


def adams_method(
    f, initial_x, initial_y, target_x, num_steps, tolerance, exact_solution
):
    print("Adams method:")

    step_size = (target_x - initial_x) / num_steps
    x_values = [initial_x]
    y_values = [initial_y]
    derivative_values = [f(initial_x, initial_y)]  # f0

    # Use Runge-Kutta to generate the first 4 points
    for i in range(3):
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
        derivative_values.append(f(next_x, next_y))  # f1, f2, f3

    if num_steps <= 3:
        print(
            "\nAdams method requires at least 4 points. Displaying Runge-Kutta results."
        )
        print_results(
            x_values, y_values, exact_solution, initial_x, initial_y, tolerance
        )
        return x_values, y_values

    # Main Adams predictor-corrector loop
    for i in range(3, num_steps):
        current_x = x_values[i]
        current_y = y_values[i]

        # Predictor step
        y_pred = adams_predictor(
            current_y,
            step_size,
            derivative_values[i],
            derivative_values[i - 1],
            derivative_values[i - 2],
            derivative_values[i - 3],
        )

        x_next = current_x + step_size
        f_pred = f(x_next, y_pred)

        # Initial corrector step
        y_corr = adams_corrector_step(
            current_y,
            step_size,
            f_pred,
            derivative_values[i],
            derivative_values[i - 1],
            derivative_values[i - 2],
        )

        # Iterate until convergence
        while abs(y_corr - y_pred) >= tolerance:
            y_pred = y_corr
            f_pred = f(x_next, y_pred)
            y_corr = adams_corrector_step(
                current_y,
                step_size,
                f_pred,
                derivative_values[i],
                derivative_values[i - 1],
                derivative_values[i - 2],
            )

        x_values.append(x_next)
        y_values.append(y_corr)
        derivative_values.append(f(x_next, y_corr))

    print("\nAdams predictor-corrector results:")
    print("x\t\tApproximate y\t\tExact y\t\t\tAbsolute Error")
    for i in range(len(x_values)):
        exact = exact_solution(x_values[i], initial_x, initial_y)
        error = abs(y_values[i] - exact)
        print(f"{x_values[i]:.6f}\t{y_values[i]:.12f}\t{exact:.12f}\t{error:.2e}")

    final_error = abs(y_values[-1] - exact_solution(target_x, initial_x, initial_y))
    if final_error < tolerance:
        print(
            f"\nAccuracy achieved: |approx_y_n - exact_y_n| = {final_error:.2e} < ε = {tolerance}"
        )
    else:
        print(
            f"\nAccuracy NOT achieved: |approx_y_n - exact_y_n| = {final_error:.2e} >= ε = {tolerance}"
        )

    # Compute exact solutions at each grid point
    exact_ys = [exact_solution(x, initial_x, initial_y) for x in x_values]
    # Calculate absolute errors
    errors = [abs(exact_y - y) for exact_y, y in zip(exact_ys, y_values)]

    print(f"max_i |y - y_true| = {max(errors)}")

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
