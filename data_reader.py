import math


def choose_odu():
    import math

    print("Выберите ОДУ:")
    print("1. y' = x + y")
    print("2. y' = sin(x) - y")
    print("3. y' = y / x")

    while True:
        try:
            choice = int(input("> Выберите ОДУ [1/2/3]: "))
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
                print("Некорректный ввод! Попробуйте снова")
        except ValueError:
            print("Некорректный ввод! Введите число")


def read_data():
    while True:
        try:
            x0 = float(input("Введите первый элемент интервала x0: "))
            xn = float(input("Введите последний элемент интервала xn: "))
            n = int(input("Введите количество элементов в интервале n: "))
            y0 = float(input("Введите y0: "))
            epsilon = float(input("Введите точность epsilon: "))
            break
        except:
            print("Некорректный ввод, попробуем еще раз!")
    return x0, xn, n, y0, epsilon
