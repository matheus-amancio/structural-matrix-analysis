import numpy as np

import pathlib


class Truss_2D:
    def __init__(self):
        pass

    def read_input_file(self, file):
        with open(file, "r", encoding="utf-8") as inp_file:
            lines = inp_file.read().split("\n")
        print(lines)


if __name__ == "__main__":
    data_path = pathlib.Path(__file__).parent / "data" / "truss2d_input_file.txt"
    trelica = Truss_2D()
    trelica.read_input_file(data_path)
