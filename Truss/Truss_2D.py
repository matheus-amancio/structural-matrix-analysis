import pathlib
import numpy as np


class Truss_2D:
    def __init__(self, input_file):
        """
        Método construtor da classe. Lê o arquivo de entrada e monta/calcula as
        matrizes necessárias para a análise.

        Parâmetros

            input_file: Caminho do arquivo com os dados de entrada.
        """
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
        lines = list(filter(lambda line: lines.index(
            line) >= self.num_nodes, lines))
        # Armazena a quantidade de apoios
        self.num_supports = int(lines.pop(0))
        self.matrix_supports = np.zeros([self.num_supports, 3], dtype=int)
        # Armazena os dados dos apoios (suportes)
        for support in range(self.num_supports):
            node, x_restrained, y_restrained = lines[support].split(",")
            self.matrix_supports[support, 0] = float(node)
            self.matrix_supports[support, 1] = float(x_restrained)
            self.matrix_supports[support, 2] = float(y_restrained)
        # Exclui os dados já armazenados da lista
        lines = list(filter(lambda line: lines.index(
            line) >= self.num_supports, lines))
        # Armazena a quantidade de módulos de elasticidade longitudinal
        self.num_E = int(lines.pop(0))
        self.matrix_E = np.zeros([self.num_E])
        # Armazena os valores dos módulos de elasticidade longitudinal
        for modulus in range(self.num_E):
            self.matrix_E[modulus] = float(lines[modulus])
        # Exclui os dados já armazenados da lista
        lines = list(
            filter(lambda line: lines.index(line) >= self.num_E, lines))
        # Armazena a quantidade de áreas de seção transversal
        self.num_areas = int(lines.pop(0))
        self.matrix_areas = np.zeros([self.num_areas])
        # Armazena os valores das áreas
        for area in range(self.num_areas):
            self.matrix_areas[area] = float(lines[area])
        # Exclui os dados já armazenados da lista
        lines = list(filter(lambda line: lines.index(
            line) >= self.num_areas, lines))
        # Armazena o número de barras da treliça
        self.num_members = int(lines.pop(0))
        self.matrix_members = np.zeros([self.num_members, 4], dtype=int)
        # Armazena as informações das barras
        for member in range(self.num_members):
            initial_node, final_node, e_type, area_type = lines[member].split(
                ",")
            self.matrix_members[member, 0] = float(initial_node)
            self.matrix_members[member, 1] = float(final_node)
            self.matrix_members[member, 2] = float(e_type)
            self.matrix_members[member, 3] = float(area_type)
        # Exclui os dados já armazenados da lista
        lines = list(filter(lambda line: lines.index(
            line) >= self.num_members, lines))
        # Armazena o número de forças aplicadas
        self.num_forces = int(lines.pop(0))
        self.matrix_forces_location = np.zeros([self.num_forces], dtype=int)
        self.matrix_forces_magnitude = np.zeros([self.num_forces, 2])
        # Armazena os nós em que as forças são aplicadas e as magnitudes das forças
        for force in range(self.num_forces):
            node, force_x, force_y = lines[force].split(",")
            self.matrix_forces_location[force] = float(node)
            self.matrix_forces_magnitude[force, 0] = float(force_x)
            self.matrix_forces_magnitude[force, 1] = float(force_y)
        # PRINTS
        # print(self.matrix_node_coords)
        # print(self.matrix_supports)
        # print(self.matrix_E)
        # print(self.matrix_areas)
        # print(self.matrix_members)
        # print(self.matrix_forces_location)
        # print(self.matrix_forces_magnitude)
        # Cálculo das matrizes necessárias para a análise
        self.calculate_num_dof()
        self.get_structure_coord_numbers()
        self.calculate_structure_stiffness_matrix()
        self.assembly_node_load_vector()

    def calculate_num_dof(self):
        """Calcula o número de graus de liberdade da estrutura."""
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
        self.num_dof = 2 * self.num_nodes - self.num_restrained_coord

    def get_structure_coord_numbers(self):
        """
        Monta a matriz com os números associados às coordenadas dos nós da
        estrutura.
        """
        # Valores auxiliares
        num_dof = self.num_dof
        matrix_supports = self.matrix_supports
        # Montagem da matriz coluna dos números das coordenadas
        self.structure_coord_numbers = np.zeros(
            [2 * self.num_nodes], dtype=int)
        # Contadores auxiliares
        count_dof = 0
        count_restrained = num_dof
        # Percorre os nós da estrutura
        for node in range(self.num_nodes):
            is_support = False
            # Percorre a matriz dos apoios
            for support_node in range(len(matrix_supports)):
                # Verifica se o nó avaliado contém um apoio
                if matrix_supports[support_node, 0] == node:
                    is_support = True
                    # Percorre as direções (x = 0, y = 1)
                    for direction in range(2):
                        coord_number = 2 * node + direction
                        # Verifica se o apoio está na direção avaliada
                        if matrix_supports[support_node, direction + 1] == 1:
                            count_restrained += 1
                            self.structure_coord_numbers[coord_number] = count_restrained - 1
                        else:
                            count_dof += 1
                            self.structure_coord_numbers[coord_number] = count_dof - 1
            if not is_support:
                for direction in range(2):
                    coord_number = 2 * node + direction
                    count_dof += 1
                    self.structure_coord_numbers[coord_number] = count_dof - 1

    def calculate_structure_stiffness_matrix(self):
        """Calcula a matriz de rigidez da ESTRUTURA."""
        # Valores auxiliares
        num_dof = self.num_dof
        matrix_members = self.matrix_members
        # Gerando as matrizes com zeros
        K = np.zeros([4, 4])
        S = np.zeros([num_dof, num_dof])
        for member in matrix_members:
            # Informações das barras
            beg_node = member[0]
            end_node = member[1]
            material_type = member[2]
            cross_section_type = member[3]
            E = self.matrix_E[material_type]
            A = self.matrix_areas[cross_section_type]
            x_beg = self.matrix_node_coords[beg_node, 0]
            y_beg = self.matrix_node_coords[beg_node, 1]
            x_end = self.matrix_node_coords[end_node, 0]
            y_end = self.matrix_node_coords[end_node, 1]
            L = np.sqrt((x_end-x_beg)**2 + (y_end-y_beg)**2)
            # cs = cos e sn = sen
            cs = (x_end-x_beg) / L
            sn = (y_end-y_beg) / L
            # Montagem da matriz de rigidez do elemento no SCG
            self.assembly_K(E, A, L, cs, sn, K)
            self.assembly_S(beg_node, end_node, K, S)
        # print(f"Structure Stiffness Matrix:\n{S}\n")
        self.structure_stiffness_matrix = S

    def assembly_K(self, E, A, L, cs, sn, K):
        """
        Preenche a matriz de rigidez do ELEMENTO (K) a partir dos parâmetros

        Parâmetros:

            E: Módulo de elasticidade longitudinal
            A: Área da seção transversal da barra
            L: Comprimento da barra
            cs: Cosseno (matriz de rotação)
            sn: Seno (matriz de rotação)
            K: Matriz de rigidez do elemento a ser preenchida
        """
        axial_stiffness = E * A / (12*L)
        # z1, z2 e z3 representam os produtos entre a rigidez axial e os senos e cossenos
        z1 = axial_stiffness * cs ** 2
        z2 = axial_stiffness * sn ** 2
        z3 = axial_stiffness * cs * sn
        K[0, 0] = z1
        K[1, 0] = z3
        K[2, 0] = -z1
        K[3, 0] = -z3
        K[0, 1] = z3
        K[1, 1] = z2
        K[2, 1] = -z3
        K[3, 1] = -z2
        K[0, 2] = -z1
        K[1, 2] = -z3
        K[2, 2] = z1
        K[3, 2] = z3
        K[0, 3] = -z3
        K[1, 3] = -z2
        K[2, 3] = z3
        K[3, 3] = z2

    def assembly_S(self, beg_node, end_node, K, S):
        """
        Preenche a matriz de rigidez da ESTRUTURA
        
        Parâmetros:

            beg_node: Nó inicial do elemento
            end_node: Nó final do elemento
            K: Matriz de rigidez do elemento no SCG
            S: Matriz de rigidez da estrutura a ser preenchida
        """
        num_dof = self.num_dof
        coord_numbers = self.structure_coord_numbers
        for row_K in range(4):
            if row_K < 2:
                row_code_number_1 = 2 * beg_node + row_K
            else:
                row_code_number_1 = 2 * end_node + (row_K - 2)
            row_S = coord_numbers[row_code_number_1]
            if row_S < num_dof:
                for column_K in range(4):
                    if column_K < 2:
                        row_code_number_2 = 2 * beg_node + column_K
                    else:
                        row_code_number_2 = 2 * end_node + (column_K - 2)
                    column_S = coord_numbers[row_code_number_2]
                    if column_S < num_dof:
                        S[row_S, column_S] += K[row_K, column_K]

    def assembly_node_load_vector(self):
        """Monta o vetor de carregamentos nodais."""
        num_dof = self.num_dof
        self.node_load_vector = np.zeros([num_dof])
        coord_numbers = self.structure_coord_numbers
        for row in range(len(self.matrix_forces_location)):
            node = self.matrix_forces_location[row]
            row_coord_number = 2 * node
            N1 = coord_numbers[row_coord_number]
            N2 = coord_numbers[row_coord_number + 1]
            if N1 < num_dof:
                self.node_load_vector[N1] += self.matrix_forces_magnitude[row, 0]
            if N2 < num_dof:
                self.node_load_vector[N2] += self.matrix_forces_magnitude[row, 1]
    
    def solve_node_displacements(self):
        """Calcula os deslocamentos nodais."""
        self.node_displacements = np.linalg.solve(self.structure_stiffness_matrix, self.node_load_vector)
        print(self.structure_stiffness_matrix)
        print(self.node_load_vector)
        print(self.node_displacements)


if __name__ == "__main__":
    # data_path = pathlib.Path(__file__).parent / "data/truss2d_input_file.txt"
    data_path = pathlib.Path(__file__).parent / "data/exemplo_livro.txt"
    trelica = Truss_2D(data_path)
    trelica.solve_node_displacements()
    # print(trelica.structure_coord_numbers)
    # print(trelica.structure_stiffness_matrix)
    # trelica.write_node_load_vector()
    # print(trelica.node_load_vector)
