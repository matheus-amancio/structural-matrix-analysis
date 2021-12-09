"""
Glossário:

SCG = sistema de coordenadas global
SCL = sistema de coordenadas local

K = matriz de rigidez do elemento no SCG
k = matriz de rigidez do elemento no SCL

v = deslocamentos nodais do elemento no SCG
u = deslocamento nodais do elemento no SCL

F = forças nodais do elemento no SCG
Q = forças nodais do elemento no SCL

P = forças globais da estrutura
S = matriz de rigidez global da estrutura
d = deslocamentos globais nodais
"""

# Importação das bibliotecas utilizadas
import pathlib
import numpy as np


class Truss_2D:
    def __init__(self, input_file):
        """
        Método construtor da classe. Lê o arquivo de entrada e monta/calcula as
        matrizes necessárias para a análise.

        Parâmetros

            [input_file]: Caminho do arquivo com os dados de entrada.
        """
        self.file_name = input_file
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
            self.matrix_supports[support, 0] = int(node) - 1
            self.matrix_supports[support, 1] = int(x_restrained)
            self.matrix_supports[support, 2] = int(y_restrained)
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
            self.matrix_members[member, 0] = int(initial_node) - 1
            self.matrix_members[member, 1] = int(final_node) - 1
            self.matrix_members[member, 2] = int(e_type) - 1
            self.matrix_members[member, 3] = int(area_type) - 1
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
            self.matrix_forces_location[force] = int(node) - 1
            self.matrix_forces_magnitude[force, 0] = float(force_x)
            self.matrix_forces_magnitude[force, 1] = float(force_y)

        # Determinação das matrizes necessárias para a análise
        self.get_structure_coord_numbers()
        self.calculate_structure_stiffness_matrix()
        self.assembly_node_load_vector()

    def get_structure_coord_numbers(self):
        """
        Monta a matriz com os números associados às coordenadas 
        dos nós da estrutura.
        """

        # Montagem da matriz coluna dos números das coordenadas
        self.structure_coord_numbers = np.zeros([2*self.num_nodes], dtype=int)

        # Percorre os nós da estrutura
        for i in range(self.num_nodes):
            for j in range(2):
                self.structure_coord_numbers[2*i + j] = 2*i + j

    def calculate_structure_stiffness_matrix(self):
        """Calcula a matriz de rigidez da ESTRUTURA."""

        # Valores auxiliares
        matrix_members = self.matrix_members

        # Gerando as matrizes com zeros
        K = np.zeros([4, 4])
        S = np.zeros([2 * self.num_nodes, 2 * self.num_nodes])

        # Percorrendo todas as barras da estrutura
        for member in matrix_members:
            # Informações das barras
            # node_i = nó inicial da barra
            # node_f = nó final da barra
            node_i = member[0]
            node_f = member[1]
            material_type = member[2]
            cross_section_type = member[3]

            # Dados obtidos após a leitura da matriz de membros
            E = self.matrix_E[material_type]
            A = self.matrix_areas[cross_section_type]
            x_i = self.matrix_node_coords[node_i, 0]
            y_i = self.matrix_node_coords[node_i, 1]
            x_f = self.matrix_node_coords[node_f, 0]
            y_f = self.matrix_node_coords[node_f, 1]

            # Cálculo do comprimento da barra
            L = np.sqrt((x_f-x_i)**2 + (y_f-y_i)**2)
            # cs = cos e sn = sen
            cs = (x_f-x_i) / L
            sn = (y_f-y_i) / L

            # Montagem da matriz de rigidez do elemento no SCG (triplo produto matricial)
            self.assembly_K(E, A, L, cs, sn, K)
            # Montagem da matriz de rigidez da estrutura excluindo as coordenadas restringidas
            self.assembly_S(node_i, node_f, K, S)

        # Atribuição da matriz de rigidez da estrutura como atributo do objeto
        self.structure_stiffness_matrix = S

    def assembly_K(self, E, A, L, cs, sn, K):
        """
        Preenche a matriz de rigidez do ELEMENTO (K) a partir dos parâmetros
        recebidos e efetuando o triplo produto matricial

        Parâmetros:

            [E]: Módulo de elasticidade longitudinal
            [A]: Área da seção transversal da barra
            [L]: Comprimento da barra
            [cs]: Cosseno (matriz de rotação)
            [sn]: Seno (matriz de rotação)
            [K]: Matriz de rigidez do elemento a ser preenchida
        """
        axial_stiffness = E * A / (L)
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

    def assembly_S(self, node_i, node_f, K, S):
        """
        Preenche a matriz de rigidez da ESTRUTURA excluindo as linhas e colunas
        associadas a coordenadas restringidas pelos apoios.

        Parâmetros:

            [node_i]: Nó inicial do elemento
            [node_f]: Nó final do elemento
            [K]: Matriz de rigidez do elemento no SCG
            [S]: Matriz de rigidez da estrutura a ser preenchida
        """

        # Valores auxiliares
        coord_numbers = self.structure_coord_numbers

        # Percorre as linhas da matriz de rigidez do elemento
        for row_K in range(4):
            # Verifica se a linha está associada ao nó inicial ou final
            if row_K < 2:
                row_code_number_1 = 2 * node_i + row_K
            else:
                row_code_number_1 = 2 * node_f + (row_K - 2)
            row_S = coord_numbers[row_code_number_1]
            # Percorre as colunas da matriz de rigidez do elemento
            for column_K in range(4):
                if column_K < 2:
                    row_code_number_2 = 2 * node_i + column_K
                else:
                    row_code_number_2 = 2 * node_f + (column_K - 2)
                column_S = coord_numbers[row_code_number_2]
                # Adiciona o valor pertinente à matriz de rigidez da estrutura
                S[row_S, column_S] += K[row_K, column_K]

    def assembly_node_load_vector(self):
        """Monta o vetor de carregamentos nodais."""

        # Valores auxiliares
        coord_numbers = self.structure_coord_numbers

        # Preenchendo o vetor desejado com zeros
        self.node_load_vector = np.zeros([2 * self.num_nodes])

        # Percorre as linhas com os nós de aplicações de forças
        for row in range(self.num_forces):
            node = self.matrix_forces_location[row]
            row_coord_number = 2 * node
            N1 = coord_numbers[row_coord_number]
            N2 = coord_numbers[row_coord_number + 1]
            self.node_load_vector[N1] += self.matrix_forces_magnitude[row, 0]
            self.node_load_vector[N2] += self.matrix_forces_magnitude[row, 1]

    def solve_node_displacements(self):
        """Calcula os deslocamentos nodais pelo método Pênalti."""
        coord_numbers = self.structure_coord_numbers
        matrix_supports = self.matrix_supports

        _S = np.copy(self.structure_stiffness_matrix)
        _P = np.copy(self.node_load_vector)

        for support in range(self.num_supports):
            node = matrix_supports[support, 0]
            for i in range(2):
                row_coord_number = 2 * node + i
                N = coord_numbers[row_coord_number]
                if matrix_supports[support, i + 1] == 1:
                    for row_S in range(2 * self.num_nodes):
                        for column_S in range(2 * self.num_nodes):
                            if row_S == N or column_S == N:
                                _S[row_S, column_S] = 0
                    _S[N, N] = 1
                    _P[N] = 0

        node_displacements = np.linalg.solve(_S, _P)
        return node_displacements

    def calculate_member_forces(self):
        """Calcula os esforços normais nas barras."""

        # Valores auxiliares
        matrix_members = self.matrix_members

        # Vetores que armazenam esforços nas barras e reações de apoio
        self.member_forces = np.empty((0, 4))
        reaction_vector = np.zeros([2*self.num_nodes])

        # Para cada elemento
        for member in matrix_members:
            # Informações das barras
            node_i = member[0]
            node_f = member[1]
            material_type = member[2]
            cross_section_type = member[3]
            E = self.matrix_E[material_type]
            A = self.matrix_areas[cross_section_type]
            x_i = self.matrix_node_coords[node_i, 0]
            y_i = self.matrix_node_coords[node_i, 1]
            x_f = self.matrix_node_coords[node_f, 0]
            y_f = self.matrix_node_coords[node_f, 1]
            L = np.sqrt((x_f-x_i)**2 + (y_f-y_i)**2)
            # cs = cos e sn = sen
            cs = (x_f-x_i) / L
            sn = (y_f-y_i) / L

            # Matriz de transformação de rotação
            T = self.assembly_T(cs, sn)

            # Matriz de deslocamentos nodais do elemento no SCG
            v = self.assembly_v(node_i, node_f)
            # Matriz de deslocamentos nodais do elemento no SCL
            u = np.dot(T, v)

            # Matriz de rigidez do elemento
            k = self.assembly_k(E, A, L)

            # Vetor de forças de extremidade do elemento no SCL
            Q = np.dot(k, u)
            self.member_forces = np.append(
                self.member_forces, np.array([Q]), axis=0)

            # Vetor de forças de extremidade do elemento no SCG
            F = np.dot(T.T, Q)
            # Preenchimento das reações de apoio
            self.assembly_reaction_vector(
                node_i, node_f, F, reaction_vector)

        self.support_reactions = reaction_vector

    def assembly_T(self, cs, sn):
        """
        Preenche a matriz de transformação de coordenadas.

        Parâmetros:
            [cs]: Cosseno (matriz de rotação)
            [sn]: Seno (matriz de rotação)
        """
        T = np.zeros([4, 4])
        T[0, 0] = cs
        T[0, 1] = sn
        T[1, 0] = -sn
        T[1, 1] = cs
        T[2, 2] = cs
        T[2, 3] = sn
        T[3, 2] = -sn
        T[3, 3] = cs
        return T

    def assembly_v(self, node_i, node_f):
        """
        Preenche o vetor de deslocamentos nodais no SCL

        Parâmetros:

            [node_i]: Nó inicial do elemento
            [node_f]: Nó final do elemento
        """
        # Valores auxiliares
        node_displacements = self.solve_node_displacements()
        coord_numbers = self.structure_coord_numbers

        # Criação do vetor com todos os elementos iguais a zero
        v = np.zeros([4])

        # Percorre as coordenadas do nó inicial e verifica se o deslocamento é livre
        for i in range(2):
            coord_number = coord_numbers[2*node_i + i]
            v[i] = node_displacements[coord_number]

        # Percorre as coordenadas do nó final e verifica se o deslocamento é livre
        for j in range(2, 4):
            coord_number = coord_numbers[2*node_f + (j-2)]
            v[j] = node_displacements[coord_number]

        return v

    def assembly_k(self, E, A, L):
        """
        Preenche a matriz de rigidez do elemento no SCL.

        Parâmetros:

            [E]: Módulo de elasticidade longitudinal
            [A]: Área da seção transversal da barra
            [L]: Comprimento da barra
        """
        k = np.zeros([4, 4])
        axial_stiffness = E * A / (L)
        k[0, 0] = axial_stiffness
        k[0, 2] = -axial_stiffness
        k[2, 0] = -axial_stiffness
        k[2, 2] = axial_stiffness
        return k

    def assembly_reaction_vector(self, node_i, node_f, F, reaction_vector):
        """
        Preenche o vetor de reações de apoio.

        Parâmetros:

            [node_i]: Nó inicial do elemento
            [node_f]: Nó final do elemento
            [F]: Vetor de forças nas extremidades do elemento no SCG
            [reaction_vector]: Vetor de reações de apoio a ser preenchido
        """

        # Valores auxiliares
        coord_numbers = self.structure_coord_numbers
        matrix_supports = self.matrix_supports

        for support in range(self.num_supports):
            if node_i == matrix_supports[support, 0]:
                # Percorre as coordenadas do nó inicial e verifica se o deslocamento é livre
                for i in range(2):
                    coord_number = coord_numbers[2*node_i + i]
                    if matrix_supports[support, i+1]:
                        reaction_vector[coord_number] += F[i]
            if node_f == matrix_supports[support, 0]:
                # Percorre as coordenadas do nó final e verifica se o deslocamento é livre
                for i in range(2, 4):
                    coord_number = coord_numbers[2*node_f + (i-2)]
                    if matrix_supports[support, i+1-2]:
                        reaction_vector[coord_number] += F[i]

    def show_results(self, save_file=False, output_file=""):
        """
        Exibe e salva os resultados da análise.

        Parâmetros:

            [save_file]: Salva os resultados em um arquivo de saída ou não
            [output_file]: Nome do arquivo de saída de resultados
        """

        # Nome do caso analisado
        case = self.file_name.with_suffix("").name
        if output_file == "":
            output_file = f"Resultados - {case}.txt"

        output_msg = "".center(75, '#') + '\n'
        output_msg += " INÍCIO DA ANÁLISE ".center(75, '#') + '\n'
        output_msg += "".center(75, '#') + '\n'
        output_msg += "".center(75, '-') + '\n'

        output_msg += f"Caso analisado: {case}\n"

        output_msg += "".center(75, '-') + '\n'

        # Informações gerais do caso
        output_msg += f"Número de nós: {self.num_nodes}\n"
        output_msg += f"Número de barras: {self.num_members}\n"
        output_msg += f"Número de barras: {self.num_members}\n"

        output_msg += "".center(75, '-') + '\n'

        # Coordenadas dos nós
        output_msg += " Nós ".center(75, '-') + '\n'
        output_msg += f"{'Nó'.center(4)} | {'Coordenada X'.center(14)} | {'Coordenada Y'.center(14)}\n"

        for i, node in enumerate(self.matrix_node_coords):
            node_number = f"{i + 1}".center(4)
            x = f"{node[0]:5.4E}".center(14)
            y = f"{node[1]:5.4E}".center(14)
            output_msg += ' | '.join([node_number, x, y]) + '\n'

        output_msg += "".center(75, '-') + '\n'

        # Barras da treliça
        output_msg += " Barras ".center(75, '-') + '\n'
        output_msg += f"{'Barra'.center(8)} | {'Nó inicial'.center(12)} | {'Nó final'.center(10)} | {'E'.center(10)} | {'A'.center(10)}\n"

        for i, member in enumerate(self.matrix_members):
            num_member = f"{i + 1}".center(8)
            node_i = f"{member[0] + 1}".center(12)
            node_f = f"{member[1] + 1}".center(10)
            material_type = member[2]
            cross_section_type = member[3]
            E = f"{self.matrix_E[material_type]:5.4E}".center(10)
            A = f"{self.matrix_areas[cross_section_type]:5.4E}".center(10)
            output_msg += ' | '.join([num_member,
                                     node_i, node_f, E, A]) + '\n'

        output_msg += "".center(75, '-') + '\n'

        # Apoios
        output_msg += " Apoios ".center(75, '-') + '\n'
        output_msg += f"{'Nó'.center(4)} | {'Restringe X'.center(13)} | {'Restringe Y'.center(13)}\n"

        for support in self.matrix_supports:
            node = f"{support[0] + 1}".center(4)
            x_has_support = "Sim".center(
                13) if support[1] else "Não".center(13)
            y_has_support = "Sim".center(
                13) if support[2] else "Não".center(13)
            output_msg += ' | '.join([node, x_has_support,
                                     y_has_support]) + '\n'

        output_msg += "".center(75, '-') + '\n'

        # Forças nodais
        output_msg += " Forças nodais ".center(75, '-') + '\n'
        output_msg += f"{'Nó'.center(4)} | {'Componente X'.center(14)} | {'Componente Y'.center(14)}\n"

        for i, node in enumerate(self.matrix_forces_location):
            node = f"{node + 1}".center(4)
            x_force = f"{self.matrix_forces_magnitude[i, 0]:5.4E}".center(14)
            y_force = f"{self.matrix_forces_magnitude[i, 1]:5.4E}".center(14)
            output_msg += ' | '.join([node, x_force, y_force]) + '\n'

        # Cálculo dos esforços normais nas barras
        self.calculate_member_forces()

        # Reações de apoio
        reactions = np.copy(self.support_reactions)
        supports = self.matrix_supports
        reaction_matrix = np.zeros([self.num_supports, 3])
        coord_numbers = self.structure_coord_numbers

        for support in range(self.num_supports):
            node = supports[support, 0]
            reaction_matrix[support, 0] = node
            for i in range(2):
                coord_number = coord_numbers[2*node + i]
                if supports[support, i+1]:
                    reaction_matrix[support, i+1] = reactions[coord_number]

        output_msg += "".center(75, '-') + '\n'

        output_msg += " Reações de apoio ".center(75, '-') + '\n'
        output_msg += f"{'Nó'.center(4)} | {'Componente X'.center(14)} | {'Componente Y'.center(14)}\n"

        for node in reaction_matrix:
            num_node = f"{int(node[0] + 1)}".center(4)
            x_reaction = f"{node[1]:5.4E}".center(14)
            y_reaction = f"{node[2]:5.4E}".center(14)
            output_msg += ' | '.join([num_node, x_reaction, y_reaction]) + '\n'

        # Deslocamentos nodais
        node_displacements = list(np.copy(self.solve_node_displacements()))
        node_displacements_matrix = np.zeros([self.num_nodes, 3])

        for i in range(self.num_nodes):
            node_displacements_matrix[i, 0] = i

        for i in range(self.num_nodes):
            for j in range(2):
                node_displacements_matrix[i, j+1] = node_displacements.pop(0)

        output_msg += "".center(75, '-') + '\n'

        output_msg += " Deslocamentos nodais ".center(75, '-') + '\n'
        output_msg += f"{'Nó'.center(4)} | {'Direção X'.center(18)} | {'Direção Y'.center(18)}\n"

        for node in node_displacements_matrix:
            num_node = f"{int(node[0] + 1)}".center(4)
            x_direction = f"{node[1]:5.4E}".center(18)
            y_direction = f"{node[2]:5.4E}".center(18)
            output_msg += ' | '.join([num_node,
                                     x_direction, y_direction]) + '\n'

        output_msg += "".center(75, '-') + '\n'

        # Esforços normais nas barras
        output_msg += " Esforço normal ".center(75, '-') + '\n'
        output_msg += f"{'Barra'.center(8)} | {'Esforço normal'.center(20)}\n"

        for member in range(self.num_members):
            num_member = f"{member + 1}".center(8)
            if self.member_forces[member, 0] > 0:
                axial_force = f"{abs(self.member_forces[member, 0]):5.4E} (Compressão)".center(
                    20)
            else:
                axial_force = f"{abs(self.member_forces[member, 0]):5.4E} (Tração)".center(
                    20)
            output_msg += ' | '.join([num_member, axial_force]) + '\n'

        output_msg += "".center(75, '-') + '\n'
        output_msg += "".center(75, '#') + '\n'
        output_msg += " FIM DA ANÁLISE ".center(75, '#') + '\n'
        output_msg += "".center(75, '#')

        # Salvamento dos resultados em um arquivo txt se save_file=True
        if save_file:
            with open(output_file, "w", encoding="utf-8") as out_file:
                out_file.write(output_msg)

        # Exibição dos resultados
        print(output_msg)


if __name__ == "__main__":
    # Caminhos para os arquivos de entrada de dados
    data_path = pathlib.Path(__file__).parent / "data/Exemplo Livro.txt"
    data_path_2 = pathlib.Path(__file__).parent / "data/Exemplo Lista Teoria 2.txt"
    # Criação da instância de um objeto e cálculo
    trelica_1 = Truss_2D(data_path)
    trelica_1.show_results(save_file=True)
    trelica_2 = Truss_2D(data_path_2)
    trelica_2.show_results(save_file=True)
