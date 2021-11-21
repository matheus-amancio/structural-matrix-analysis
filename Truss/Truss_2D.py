import pathlib


import numpy as np


class Truss_2D:
    def __init__(self, input_file):
        # Leitura do arquivo de entrada
        with open(input_file, "r", encoding="utf-8") as inp_file:
            lines = inp_file.read().split("\n")
        # Armazena a quantidade de nós
        self.num_nodes = int(lines.pop(0))
        self.matrix_node_coords = np.zeros([self.num_nodes, 2])
        # Armazena as coordenadas dos nós
        for node in range(self.num_nodes):
            x, y = lines[node].split(",")
            self.matrix_node_coords[node, 0] = float(x)
            self.matrix_node_coords[node, 1] = float(y)
        # Exclui os dados já armazenados da lista
        lines = list(filter(lambda line: lines.index(line) >= self.num_nodes, lines))
        # Armazena a quantidade de apoios
        self.num_supports = int(lines.pop(0))
        self.matrix_supports = np.zeros([self.num_supports, 3])
        # Armazena os dados dos apoios (suportes)
        for support in range(self.num_supports):
            node, x_restrained, y_restrained = lines[support].split(",")
            self.matrix_supports[support, 0] = float(node)
            self.matrix_supports[support, 1] = float(x_restrained)
            self.matrix_supports[support, 2] = float(y_restrained)
        # Exclui os dados já armazenados da lista
        lines = list(filter(lambda line: lines.index(line) >= self.num_supports, lines))
        # Armazena a quantidade de módulos de elasticidade longitudinal
        self.num_E = int(lines.pop(0))
        self.matrix_E = np.zeros([self.num_E, 1])
        # Armazena os valores dos módulos de elasticidade longitudinal
        for modulus in range(self.num_E):
            self.matrix_E[modulus, 0] = float(lines[modulus])
        # Exclui os dados já armazenados da lista
        lines = list(filter(lambda line: lines.index(line) >= self.num_E, lines))
        # Armazena a quantidade de áreas de seção transversal
        self.num_areas = int(lines.pop(0))
        self.matrix_areas = np.zeros([self.num_areas, 1])
        # Armazena os valores das áreas
        for area in range(self.num_areas):
            self.matrix_areas[area, 0] = float(lines[area])
        # Exclui os dados já armazenados da lista
        lines = list(filter(lambda line: lines.index(line) >= self.num_areas, lines))
        # Armazena o número de barras da treliça
        self.num_members = int(lines.pop(0))
        self.matrix_members = np.zeros([self.num_members, 4])
        # Armazena as informações das barras
        for member in range(self.num_members):
            initial_node, final_node, e_type, area_type = lines[member].split(",")
            self.matrix_members[member, 0] = float(initial_node)
            self.matrix_members[member, 1] = float(final_node)
            self.matrix_members[member, 2] = float(e_type)
            self.matrix_members[member, 3] = float(area_type)
        # Exclui os dados já armazenados da lista
        lines = list(filter(lambda line: lines.index(line) >= self.num_members, lines))
        # Armazena o número de forças aplicadas
        self.num_forces = int(lines.pop(0))
        self.matrix_forces_location = np.zeros([self.num_forces, 1])
        self.matrix_forces_magnitude = np.zeros([self.num_forces, 2])
        # Armazena os nós em que as forças são aplicadas e as magnitudes das forças
        for force in range(self.num_forces):
            node, force_x, force_y = lines[force].split(",")
            self.matrix_forces_location[force, 0] = float(node)
            self.matrix_forces_magnitude[force, 0] = float(force_x)
            self.matrix_forces_magnitude[force, 1] = float(force_y)

    def calculate_num_dof(self):
        # Número de coordenadas restringidas
        self.num_restrained_coord = 0
        # Percorre as linhas da matriz de suporte
        for row in range(len(self.matrix_supports)):
            # Percorre a segunda e terceira colunas da matriz de suporte
            for column in range(1, len(self.matrix_supports[0])):
                # Verifica se há apoio na direção x ou y
                if self.matrix_supports[row, column] == 1:
                    self.num_restrained_coord += 1
        # Cálculo do número de graus de liberdade
        num_dof = 2 * self.num_nodes - self.num_restrained_coord
        return num_dof




if __name__ == "__main__":
    data_path = pathlib.Path(__file__).parent / "data/truss2d_input_file.txt"
    trelica = Truss_2D(data_path)
    trelica.calculate_num_dof()
