from typing import Callable


def runge_kutt(
    f,
    initial_x,
    initial_y,
    target_x,
    initial_steps,
    epsilon,
    exact_solution: Callable[[float, float, float], float],
):
    print("Fourth-order Runge-Kutta method \n")
    start_x = initial_x
    start_y = initial_y
    max_iterations = 10
    order = 4

    original_step_count = initial_steps
    iteration_count = 0
    error_estimate = float("inf")

    solution_history = []

    while error_estimate >= epsilon and iteration_count < max_iterations:
        step_size = (target_x - initial_x) / initial_steps
        x_values = [initial_x]
        y_values = [initial_y]

        for _ in range(initial_steps):
            current_x = x_values[-1]
            current_y = y_values[-1]

            k1 = step_size * f(current_x, current_y)
            k2 = step_size * f(current_x + step_size / 2, current_y + k1 / 2)
            k3 = step_size * f(current_x + step_size / 2, current_y + k2 / 2)
            k4 = step_size * f(current_x + step_size, current_y + k3)

            next_y = current_y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            next_x = current_x + step_size

            y_values.append(next_y)
            x_values.append(next_x)

        solution_history.append(y_values[-1])

        if len(solution_history) >= 2:
            last_approximation = solution_history[-2]
            current_approximation = solution_history[-1]
            error_estimate = abs(last_approximation - current_approximation) / (
                2**order - 1
            )

            if error_estimate >= epsilon:
                print(
                    f"Precision not achieved at step_count={initial_steps}: EST = {error_estimate:.2e} ≥ ε ({epsilon})"
                )
                print(f"Doubling step count: n = {initial_steps} → {2*initial_steps}")
            else:
                break
        else:
            print(f"Doubling step count: n = {initial_steps} → {2*initial_steps}")

        initial_steps *= 2
        iteration_count += 1

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print(f"Initial step count: n₀ = {original_step_count}")
    print(f"Final step count: n = {initial_steps}")
    print(f"Step doubling iterations: {iteration_count}")

    if error_estimate < epsilon:
        print(f"Precision achieved: EST = {error_estimate:.2e} < ε = {epsilon}")
    else:
        print(f"Maximum iterations reached: EST = {error_estimate:.2e} ≥ ε = {epsilon}")

    if len(x_values) <= 100:
        print("\nSolution table:")
        print("x_value\t\tApproximate_y")
        print("-" * 80)
        for i in range(len(x_values)):
            print(f"{x_values[i]:.6f}\t{y_values[i]:.12f}")
    else:
        print("\nResults omitted in tabular form (exceeds 100 points)")
        print(f"Total points: {len(x_values)} > 100")

        print("\nKey solution points:")
        print("x_value\t\tApproximate_y")
        print("-" * 80)
        print(f"{x_values[0]:.6f}\t{y_values[0]:.12f}")
        mid_index = len(x_values) // 2
        print(f"{x_values[mid_index]:.6f}\t{y_values[mid_index]:.12f}")
        print(f"{x_values[-1]:.6f}\t{y_values[-1]:.12f}")
        print("... (intermediate points omitted)")

    # Compute exact solutions at each grid point
    exact_ys = [exact_solution(x, initial_x, initial_y) for x in x_values]
    # Calculate absolute errors
    errors = [abs(exact_y - y) for exact_y, y in zip(exact_ys, y_values)]

    print(f"max_i |y - y_true| = {max(errors)}")

    return x_values, y_values
