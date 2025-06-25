def euler_method(f, initial_x, initial_y, target_x, initial_step_count, tolerance):
    print("Euler Method\n")
    max_doubling_iterations = 100
    method_order = 1
    doubling_count = 0
    error_estimate = float("inf")

    final_y_history = []

    while error_estimate >= tolerance and doubling_count < max_doubling_iterations:
        x_values = [initial_x]
        y_values = [initial_y]

        step_size = (target_x - initial_x) / initial_step_count
        for _ in range(initial_step_count):
            current_x = x_values[-1]
            current_y = y_values[-1]
            next_y = current_y + step_size * f(current_x, current_y)
            next_x = current_x + step_size
            y_values.append(next_y)
            x_values.append(next_x)
        final_y_history.append(y_values[-1])

        if len(final_y_history) >= 2:
            y_coarse, y_fine = final_y_history[-2:]
            error_estimate = abs(y_coarse - y_fine) / (2**method_order - 1)

            if error_estimate >= tolerance:
                print(
                    f"Accuracy not achieved at step_count={initial_step_count}: R = {error_estimate:.2e} >= ε ({tolerance})"
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
    if len(x_values) >= 100:
        print("\nResults not displayed in table (too many points)")
        print(f"Point count: {len(x_values)} > 100")
        print("\nKey points:")
        print("x\t\tApproximate y")
        print("-" * 80)
        print(f"{x_values[0]:.6f}\t{y_values[0]:.12f}")

        mid_index = len(x_values) // 2
        print(f"{x_values[mid_index]:.6f}\t{y_values[mid_index]:.12f}")

        print(f"{x_values[-1]:.6f}\t{y_values[-1]:.12f}")
        print("... (intermediate points omitted)")

    else:
        print("x\t\tApproximate y")
        print("-" * 80)
        for i in range(len(x_values)):
            print(f"{x_values[i]:.6f}\t{y_values[i]:.12f}")

    return x_values, y_values
