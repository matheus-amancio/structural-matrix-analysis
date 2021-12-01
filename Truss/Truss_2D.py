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
        self.calculate_num_dof()
        self.get_structure_coord_numbers()
        self.calculate_structure_stiffness_matrix()
        self.assembly_node_load_vector()

    def calculate_num_dof(self):
        """Calcula o número de graus de liberdade da estrutura."""

        # Número de coordenadas restringidas (contador)
        self.num_restrained_coord = 0

        # Percorre as linhas da matriz de suporte
        for row in range(self.num_supports):
            # Percorre a segunda e terceira colunas da matriz de suporte
            for column in range(1, len(self.matrix_supports[0])):
                # Verifica se há apoio na direção x ou y
                if self.matrix_supports[row, column] == 1:
                    self.num_restrained_coord += 1

        # Cálculo do número de graus de liberdade
        self.num_dof = 2 * self.num_nodes - self.num_restrained_coord

    def get_structure_coord_numbers(self):
        """
        Monta a matriz com os números associados às coordenadas 
        dos nós da estrutura.

        Números de coordenadas maiores ou iguais ao número de graus
        de liberdade estão associados a coordenadas restringidas por
        apoios.
        """

        # Valores auxiliares
        num_dof = self.num_dof
        matrix_supports = self.matrix_supports

        # Montagem da matriz coluna dos números das coordenadas
        self.structure_coord_numbers = np.zeros([2*self.num_nodes], dtype=int)

        # Contadores auxiliares
        count_dof = 0
        count_restrained = num_dof

        # Percorre os nós da estrutura
        for node in range(self.num_nodes):
            has_support = False

            # Percorre a matriz dos apoios
            for support_node in range(self.num_supports):
                # Verifica se o nó avaliado contém um apoio
                if matrix_supports[support_node, 0] == node:
                    has_support = True

                    # Percorre as direções (x = 0, y = 1)
                    for i in range(2):
                        coord_number = 2*node + i
                        # Verifica se o apoio está na direção avaliada
                        # Atribui o número de coordenada (-1 pela indexação do python)
                        if matrix_supports[support_node, i + 1] == 1:
                            count_restrained += 1
                            self.structure_coord_numbers[coord_number] = count_restrained - 1
                        else:
                            count_dof += 1
                            self.structure_coord_numbers[coord_number] = count_dof - 1

            if not has_support:
                # Percorre as direções (x = 0, y = 1)
                for i in range(2):
                    coord_number = 2 * node + i
                    count_dof += 1
                    # Atribui o número de coordenada (-1 pela indexação do python)
                    self.structure_coord_numbers[coord_number] = count_dof - 1

    def calculate_structure_stiffness_matrix(self):
        """Calcula a matriz de rigidez da ESTRUTURA."""

        # Valores auxiliares
        num_dof = self.num_dof
        matrix_members = self.matrix_members

        # Gerando as matrizes com zeros
        K = np.zeros([4, 4])
        S = np.zeros([num_dof, num_dof])

        # Percorrendo todas as barras da estrutura
        for member in matrix_members:
            # Informações das barras
            # beg_node = nó inicial da barra
            # end_node = nó final da barra
            beg_node = member[0]
            end_node = member[1]
            material_type = member[2]
            cross_section_type = member[3]

            # Dados obtidos após a leitura da matriz de membros
            E = self.matrix_E[material_type]
            A = self.matrix_areas[cross_section_type]
            x_beg = self.matrix_node_coords[beg_node, 0]
            y_beg = self.matrix_node_coords[beg_node, 1]
            x_end = self.matrix_node_coords[end_node, 0]
            y_end = self.matrix_node_coords[end_node, 1]

            # Cálculo do comprimento da barra
            L = np.sqrt((x_end-x_beg)**2 + (y_end-y_beg)**2)
            # cs = cos e sn = sen
            cs = (x_end-x_beg) / L
            sn = (y_end-y_beg) / L

            # Montagem da matriz de rigidez do elemento no SCG (triplo produto matricial)
            self.assembly_K(E, A, L, cs, sn, K)
            # Montagem da matriz de rigidez da estrutura excluindo as coordenadas restringidas
            self.assembly_S(beg_node, end_node, K, S)

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

    def assembly_S(self, beg_node, end_node, K, S):
        """
        Preenche a matriz de rigidez da ESTRUTURA excluindo as linhas e colunas
        associadas a coordenadas restringidas pelos apoios.

        Parâmetros:

            [beg_node]: Nó inicial do elemento
            [end_node]: Nó final do elemento
            [K]: Matriz de rigidez do elemento no SCG
            [S]: Matriz de rigidez da estrutura a ser preenchida
        """

        # Valores auxiliares
        num_dof = self.num_dof
        coord_numbers = self.structure_coord_numbers

        # Percorre as linhas da matriz de rigidez do elemento
        for row_K in range(4):
            # Verifica se a linha está associada ao nó inicial ou final
            if row_K < 2:
                row_code_number_1 = 2 * beg_node + row_K
            else:
                row_code_number_1 = 2 * end_node + (row_K - 2)
            row_S = coord_numbers[row_code_number_1]
            # Verifica se a linha está associada a uma coordenada restringida
            if row_S < num_dof:
                # Percorre as colunas da matriz de rigidez do elemento
                for column_K in range(4):
                    if column_K < 2:
                        row_code_number_2 = 2 * beg_node + column_K
                    else:
                        row_code_number_2 = 2 * end_node + (column_K - 2)
                    column_S = coord_numbers[row_code_number_2]
                    # Verifica se a coluna está associada a uma coordenada restringida
                    if column_S < num_dof:
                        # Adiciona o valor pertinente à matriz de rigidez da estrutura
                        S[row_S, column_S] += K[row_K, column_K]

    def assembly_node_load_vector(self):
        """Monta o vetor de carregamentos nodais."""

        # Valores auxiliares
        num_dof = self.num_dof
        coord_numbers = self.structure_coord_numbers

        # Preenchendo o vetor desejado com zeros
        self.node_load_vector = np.zeros([num_dof])

        # Percorre as linhas com os nós de aplicações de forças
        for row in range(self.num_forces):
            node = self.matrix_forces_location[row]
            row_coord_number = 2 * node
            N1 = coord_numbers[row_coord_number]
            N2 = coord_numbers[row_coord_number + 1]
            # Verifica se a coordenada é restringida por algum apoio
            if N1 < num_dof:
                self.node_load_vector[N1] += self.matrix_forces_magnitude[row, 0]
            # Verifica se a coordenada é restringida por algum apoio
            if N2 < num_dof:
                self.node_load_vector[N2] += self.matrix_forces_magnitude[row, 1]

    def solve_node_displacements(self):
        """Calcula os deslocamentos nodais."""
        node_displacements = np.linalg.solve(
            self.structure_stiffness_matrix, self.node_load_vector
        )
        return node_displacements

    def calculate_member_forces(self):
        """Calcula os esforços normais nas barras."""

        # Valores auxiliares
        matrix_members = self.matrix_members

        # Vetores que armazenam esforços nas barras e reações de apoio
        self.member_forces = np.empty((0, 4))
        reaction_vector = np.zeros([self.num_restrained_coord])

        # Para cada elemento
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

            # Matriz de transformação de rotação
            T = self.assembly_T(cs, sn)

            # Matriz de deslocamentos nodais do elemento no SCG
            v = self.assembly_v(beg_node, end_node)
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
            self.assembly_reaction_vector(beg_node, end_node, F, reaction_vector)

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

    def assembly_v(self, beg_node, end_node):
        """
        Preenche o vetor de deslocamentos nodais no SCL

        Parâmetros:

            [beg_node]: Nó inicial do elemento
            [end_node]: Nó final do elemento
        """
        # Valores auxiliares
        node_displacements = self.solve_node_displacements()
        coord_numbers = self.structure_coord_numbers

        # Criação do vetor com todos os elementos iguais a zero
        v = np.zeros([4])

        # Percorre as coordenadas do nó inicial e verifica se o deslocamento é livre
        for i in range(2):
            coord_number = coord_numbers[2*beg_node + i]
            if coord_number < self.num_dof:
                v[i] = node_displacements[coord_number]

        # Percorre as coordenadas do nó final e verifica se o deslocamento é livre
        for j in range(2, 4):
            coord_number = coord_numbers[2*end_node + (j-2)]
            if coord_number < self.num_dof:
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

    def assembly_reaction_vector(self, beg_node, end_node, F, reaction_vector):
        """
        Preenche o vetor de reações de apoio.

        Parâmetros:

            [beg_node]: Nó inicial do elemento
            [end_node]: Nó final do elemento
            [F]: Vetor de forças nas extremidades do elemento no SCG
            [reaction_vector]: Vetor de reações de apoio a ser preenchido
        """

        # Valores auxiliares
        coord_numbers = self.structure_coord_numbers

        # Percorre as coordenadas do nó inicial e verifica se o deslocamento é livre
        for i in range(2):
            coord_number = coord_numbers[2*beg_node + i]
            if coord_number >= self.num_dof:
                reaction_vector[coord_number - self.num_dof] = F[i]

        # Percorre as coordenadas do nó final e verifica se o deslocamento é livre
        for j in range(2, 4):
            coord_number = coord_numbers[2*end_node + (j-2)]
            if coord_number >= self.num_dof:
                reaction_vector[coord_number - self.num_dof] = F[j]

    def show_results(self, save_file=False, output_file="results.txt"):
        """
        Exibe e salva os resultados da análise.

        Parâmetros:

            [save_file]: Salva os resultados em um arquivo de saída
            [output_file]: Nome do arquivo de saída
        """

        # Nome do caso analisado
        case = self.file_name.with_suffix("").name

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
            beg_node = f"{member[0] + 1}".center(12)
            end_node = f"{member[1] + 1}".center(10)
            material_type = member[2]
            cross_section_type = member[3]
            E = f"{self.matrix_E[material_type]:5.4E}".center(10)
            A = f"{self.matrix_areas[cross_section_type]:5.4E}".center(10)
            output_msg += ' | '.join([num_member,
                                     beg_node, end_node, E, A]) + '\n'

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
        reactions = list(np.copy(self.support_reactions))
        supports = self.matrix_supports
        reaction_matrix = np.zeros([self.num_supports, 3])

        for i in range(self.num_supports):
            reaction_matrix[i, 0] = self.matrix_supports[i, 0]
            if supports[i, 1] == 1:
                reaction_matrix[i, 1] = reactions.pop(0)
            if supports[i, 2] == 1:
                reaction_matrix[i, 2] = reactions.pop(0)

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
                coord_number_row = 2*i + j
                coord_number = self.structure_coord_numbers[coord_number_row] 
                if coord_number < self.num_dof:
                    node_displacements_matrix[i, j+1] = node_displacements.pop(0)

        output_msg += "".center(75, '-') + '\n'

        output_msg += " Deslocamentos nodais ".center(75, '-') + '\n'
        output_msg += f"{'Nó'.center(4)} | {'Direção X'.center(18)} | {'Direção Y'.center(18)}\n"

        for node in node_displacements_matrix:
            num_node = f"{int(node[0] + 1)}".center(4)
            x_direction = f"{node[1]:5.4E}".center(18)
            y_direction = f"{node[2]:5.4E}".center(18)
            output_msg += ' | '.join([num_node, x_direction, y_direction]) + '\n'

        output_msg += "".center(75, '-') + '\n'

        # Esforços normais nas barras
        output_msg += " Esforço normal ".center(75, '-') + '\n'
        output_msg += f"{'Barra'.center(8)} | {'Esforço'.center(9)}\n"

        for member in range(self.num_members):
            num_member = f"{member + 1}".center(8)
            if self.member_forces[member, 0] > 0:
                axial_force = f"{abs(self.member_forces[member, 0]):5.4E} (Compressão)".center(14)
            else:
                axial_force = f"{abs(self.member_forces[member, 0]):5.4E} (Tração)".center(14)
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
    data_path = pathlib.Path(__file__).parent / "data/truss2d_input_file.txt"
    trelica = Truss_2D(data_path)
    trelica.show_results(save_file=True)
