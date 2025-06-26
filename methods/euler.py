from typing import Callable


def euler_method(
    f,
    initial_x,
    initial_y,
    target_x,
    initial_step_count,
    tolerance,
    exact_solution: Callable[[float, float, float], float],  # x, x_0, y_0
):
    print("Euler Method\n")
    MAX_DOUBLING_ATTEMPTS = 100
    METHOD_ORDER = 1
    doubling_count = 0
    error_estimate = float("inf")

    solution_history = []

    while error_estimate >= tolerance and doubling_count < MAX_DOUBLING_ATTEMPTS:
        x_coordinates = [initial_x]
        y_values = [initial_y]

        step_size = (target_x - initial_x) / initial_step_count
        for _ in range(initial_step_count):
            current_x = x_coordinates[-1]
            current_y = y_values[-1]
            next_y = current_y + step_size * f(current_x, current_y)
            next_x = current_x + step_size
            y_values.append(next_y)
            x_coordinates.append(next_x)
        solution_history.append(y_values[-1])

        if len(solution_history) >= 2:
            previous_solution = solution_history[-2]
            current_solution = solution_history[-1]
            error_estimate = abs(current_solution - previous_solution) / (
                2**METHOD_ORDER - 1
            )

            if error_estimate >= tolerance:
                print(
                    f"Accuracy not achieved with n={initial_step_count}: R = {error_estimate:.2e} >= ε ({tolerance})"
                )
                print(
                    f"Doubling step count: {initial_step_count} → {2*initial_step_count}"
                )
            else:
                break
        else:
            print(f"Doubling step count: {initial_step_count} → {2*initial_step_count}")
        initial_step_count *= 2
        doubling_count += 1

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print(f"Final step count: n = {initial_step_count}")
    print(f"Number of doublings: {doubling_count}")

    if error_estimate < tolerance:
        print(f"Accuracy achieved: R = {error_estimate:.2e} < ε = {tolerance}")
    else:
        print(f"Doubling limit reached: R = {error_estimate:.2e} >= ε = {tolerance}")

    print("\nSolution:")
    if len(x_coordinates) >= 100:
        print("\nResults not displayed in table (too many points)")
        print(f"Point count: {len(x_coordinates)} > 100")
        print("\nKey points:")
        print("x-coordinate\tApproximate y-value")
        print("-" * 80)
        print(f"{x_coordinates[0]:.6f}\t{y_values[0]:.12f}")

        mid_index = len(x_coordinates) // 2
        print(f"{x_coordinates[mid_index]:.6f}\t{y_values[mid_index]:.12f}")

        print(f"{x_coordinates[-1]:.6f}\t{y_values[-1]:.12f}")
        print("... (intermediate points omitted)")
    else:
        print("x-coordinate\tApproximate y-value")
        print("-" * 80)
        for i in range(len(x_coordinates)):
            print(f"{x_coordinates[i]:.6f}\t{y_values[i]:.12f}")

    # Compute exact solutions at each grid point
    exact_ys = [exact_solution(x, initial_x, initial_y) for x in x_coordinates]
    # Calculate absolute errors
    errors = [abs(exact_y - y) for exact_y, y in zip(exact_ys, y_values)]

    print(f"max_i |y - y_true| = {max(errors)}")

    return x_coordinates, y_values
