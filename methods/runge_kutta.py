def runge_kutt(f, x0, y0, xn, n, epsilon):
    print("Runge-Kutta 4th order method with adaptive step size:\n")
    x0_init = x0
    y0_init = y0
    max_iterations = 10
    p = 4

    original_n = n
    iteration_count = 0
    R = float("inf")

    while R >= epsilon and iteration_count < max_iterations:
        h = (xn - x0) / n
        x_arr = [x0]
        y_arr = [y0]

        for _ in range(n):
            xi = x_arr[-1]
            yi = y_arr[-1]

            k1 = h * f(xi, yi)
            k2 = h * f(xi + h / 2, yi + k1 / 2)
            k3 = h * f(xi + h / 2, yi + k2 / 2)
            k4 = h * f(xi + h, yi + k3)

            next_y = yi + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            next_x = xi + h

            y_arr.append(next_y)
            x_arr.append(next_x)

        n2 = 2 * n
        h2 = (xn - x0) / n2
        x_arr2 = [x0]
        y_arr2 = [y0]

        for _ in range(n2):
            xi = x_arr2[-1]
            yi = y_arr2[-1]

            k1 = h2 * f(xi, yi)
            k2 = h2 * f(xi + h2 / 2, yi + k1 / 2)
            k3 = h2 * f(xi + h2 / 2, yi + k2 / 2)
            k4 = h2 * f(xi + h2, yi + k3)

            next_y = yi + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            next_x = xi + h2

            y_arr2.append(next_y)
            x_arr2.append(next_x)

        yh = y_arr[-1]
        yh2 = y_arr2[-1]
        R = abs(yh - yh2) / (2**p - 1)

        if R >= epsilon:
            print(f"Precision not achieved with n={n}: R = {R:.2e} >= ε = {epsilon}")
            print(f"Doubling steps: n = {n} → {2*n}")
            n *= 2
            iteration_count += 1

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print(f"Initial step count: n₀ = {original_n}")
    print(f"Final step count: n = {n}")
    print(f"Step doubling iterations: {iteration_count}")

    if R < epsilon:
        print(f"Precision achieved: |R| = {R:.2e} < ε = {epsilon}")
    else:
        print(f"Max iterations reached: |R| = {R:.2e} >= ε = {epsilon}")

    if len(x_arr) <= 100:
        print("\nSolution points:")
        print("x-value\t\tApproximated y-value")
        print("-" * 60)
        for i in range(len(x_arr)):
            print(f"{x_arr[i]:.6f}\t{y_arr[i]:.12f}")
    else:
        print("\nResults omitted (exceed 100 points)")
        print(f"Total points calculated: {len(x_arr)}")

        print("\nKey solution points:")
        print("x-value\t\tApproximated y-value")
        print("-" * 60)

        print(f"{x_arr[0]:.6f}\t{y_arr[0]:.12f} (start)")

        mid_index = len(x_arr) // 2
        print(f"{x_arr[mid_index]:.6f}\t{y_arr[mid_index]:.12f} (midpoint)")

        print(f"{x_arr[-1]:.6f}\t{y_arr[-1]:.12f} (endpoint)")
        print("... (intermediate points hidden)")

    return x_arr, y_arr
