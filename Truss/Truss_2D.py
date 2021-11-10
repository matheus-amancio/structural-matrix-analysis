import numpy as np

import pathlib


class Truss_2D:
    def __init__(self, file):
        with open(file, "r", encoding="utf-8") as inp_file:
            lines = inp_file.read().split("\n")
        self.num_nodes = int(lines.pop(0))
        self.node_coords = np.zeros([self.num_nodes, 2])
        for node in range(self.num_nodes):
            coords = lines[node].split(",")
            self.node_coords[node, 1] = int(coords[1])
            self.node_coords[node, 0] = int(coords[0])
        lines = list(filter(lambda line: lines.index(line) >= self.num_nodes, lines))
        print(self.node_coords)
        print(lines)


if __name__ == "__main__":
    data_path = pathlib.Path(__file__).parent / "data/truss2d_input_file.txt"
    trelica = Truss_2D(data_path)
