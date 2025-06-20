import math


def choose_odu():
    print("Choose ODE:")
    print("1. y' = x + y")
    print("2. y' = sin(x) - y")
    print("3. y' = y / x")

    while True:
        try:
            choice = int(input("> Select ODE [1/2/3]: "))
            if choice == 1:
                f = lambda x, y: x + y
                exact_y = lambda x, x0, y0: math.exp(x - x0) * (y0 + x0 + 1) - x - 1
                return f, exact_y
            elif choice == 2:
                f = lambda x, y: math.sin(x) - y
                exact_y = (
                    lambda x, x0, y0: (y0 - 0.5 * math.sin(x0) + 0.5 * math.cos(x0))
                    * math.exp(x0 - x)
                    + 0.5 * math.sin(x)
                    - 0.5 * math.cos(x)
                )
                return f, exact_y
            elif choice == 3:
                f = lambda x, y: y / x
                exact_y = lambda x, x0, y0: y0 * math.exp(math.log(x) - math.log(x0))
                return f, exact_y
            else:
                print("Invalid input! Please try again")
        except ValueError:
            print("Invalid input! Enter a number")


def read_data():
    while True:
        try:
            x0 = float(input("Enter first interval element x0: "))
            xn = float(input("Enter last interval element xn: "))
            n = int(input("Enter number of elements in interval n: "))
            y0 = float(input("Enter initial value y0: "))
            epsilon = float(input("Enter precision epsilon: "))
            break
        except:
            print("Invalid input, please try again!")
    return x0, xn, n, y0, epsilon
