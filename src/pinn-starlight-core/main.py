# This is a sample Python script.
import numpy

a = numpy.array([3,2])
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(a)
    a[numpy.arange(0), 0] = 1
    print(a)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
