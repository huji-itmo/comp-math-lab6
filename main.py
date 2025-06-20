from data_reader import choose_odu, read_data
from methods.euler import euler_method
from methods.runge_kutta import runge_kutt
from methods.adams import adams_method


def main():
    f, exact_y = choose_odu()
    x0, xn, n, y0, epsilon = read_data()
    runge_kutt(f, x0, y0, xn, n, epsilon)
    euler_method(f, x0, y0, xn, n, epsilon)
    adams_method(f, x0, y0, xn, n, epsilon, exact_y)


if __name__ == "__main__":
    main()
