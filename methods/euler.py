def euler_method(f, x0, y0, xn, n, epsilon):
    print("\nEuler's Numerical Scheme\n")
    x0_init = x0
    y0_init = y0
    max_iterations = 10
    p = 1
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
            next_y = yi + h * f(xi, yi)
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
            next_y = yi + h2 * f(xi, yi)
            next_x = xi + h2
            y_arr2.append(next_y)
            x_arr2.append(next_x)

        yh = y_arr[-1]
        yh2 = y_arr2[-1]
        R = abs(yh - yh2) / (2**p - 1)

        if R >= epsilon:
            print(f"Precision not met with n={n}: |R| = {R:.2e} >= ε = {epsilon}")
            print(f"Increasing step count: n = {n} → {2*n}")
            n *= 2
            iteration_count += 1

    print("\n" + "═" * 80)
    print("FINAL OUTCOME")
    print(f"Final step count: n = {n}")
    print(f"Step doubling operations: {iteration_count}")

    if R < epsilon:
        print(f"Required precision achieved: |R| = {R:.2e} < ε = {epsilon}")
    else:
        print(f"Doubling limit reached: |R| = {R:.2e} >= ε = {epsilon}")

    print("\nComputation summary:")
    if len(x_arr) >= 100:
        print("Solution overview (full table omitted due to size)")
        print(f"Final point → x: {x_arr[-1]:.6f}, y: {y_arr[-1]:.12f}\n")

    else:
        print("Detailed results not displayed (exceeds 100 points)")
        print(f"Total points computed: {len(x_arr)}")

        print("\nCritical data points:")
        print("x_coord\t\tApproximated_y")
        print("----------------------------------------")

        print(f"{x_arr[0]:.6f}\t{y_arr[0]:.12f}")

        if len(x_arr) > 2:
            mid_index = len(x_arr) // 2
            print(f"{x_arr[mid_index]:.6f}\t{y_arr[mid_index]:.12f}")

        print(f"{x_arr[-1]:.6f}\t{y_arr[-1]:.12f}")
        if len(x_arr) > 3:
            print("... additional points not shown ...")

    return x_arr, y_arr
