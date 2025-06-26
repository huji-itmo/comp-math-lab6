from matplotlib import pyplot as plt


def plot_graphic(X, Y, func, x0, y0, method_name):
    Y_true = [func(x, x0, y0) for x in X]

    plt.figure(figsize=(8, 5))
    plt.plot(X, Y_true, label="Точное решение", linewidth=2)
    plt.plot(X, Y, "--", label="Численное решение", linewidth=2)
    plt.scatter(X, Y, color="red", s=30)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Сравнение решений ({method_name})")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{method_name}.pdf")
    plt.close()
