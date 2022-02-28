"""Módulo que resolve pórtico plano."""

"""
GLOSSÁRIO

SCL = Sistema de coordenadas local.
SCG = Sistema de coordenadas global.

k = matriz de rigidez da barra no SCL.
K = matriz de rigidez da barra no SCG.
T = matriz de rotação da barra.

u = deslocamentos das extremidades das barras no SCL.
v = deslocamentos das extremidades das barras no SCG.

Q = esforços das extremidades das barras no SCL.
F = esforços das extremidades das barras no SCG.

S = matriz de rigidez global do pórtico.
"""


# Importação das bibliotecas necessárias
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


class Material:
    """Classe que define um material."""

    def __init__(self, id, E):
        """
        Método inicializador.

        Parâmetros:
            id (int): identificador.
            E (float): módulo de elasticidade longitudinal.
        """
        self.id = id
        self.E = E


class CrossSection:
    """Classe que define uma seção transversal."""

    def __init__(self, id, A, I):
        """
        Método inicializador.

        Parâmetros:
            A (float): área.
            I (float): momento de inércia.
        """
        self.id = id
        self.A = A
        self.I = I


class NodeLoad:
    """Classe que define uma carga nodal aplicada."""

    def __init__(self, id, node_id, Fx, Fy, Mz):
        """
        Método inicializador.

        Parâmetros:
            id (int): identificador.
            node_id (int): identificador do nó de aplicação.
            F_x (float): Força linear na direção x global (horizontal).
            F_y (float): Força linear na direção y global (vertical).
            F_z (float): Força momento em torno do eixo z.
        """
        self.id = id
        self.node_id = node_id
        self.Fx = Fx
        self.Fy = Fy
        self.Mz = Mz
        self.vector = np.array([Fx, Fy, Mz])


class MemberLoad:
    """Classe que define uma carga distribuída."""

    def __init__(self, id, load_type, *load_data):
        """
        Método inicializador.

        Parâmetros:
            id (int): identificador.
            load_type (str): tipo de carregamento distribuído.
            load_data ((float)): informações sobre o carregamento (como qx e qy).
        """
        self.id = id
        self.load_type = load_type
        if "UNIFORM" in self.load_type:
            # qx e qy estão no SCL
            self.qx, self.qy = load_data


class Support:
    """Classe que define um apoio."""

    def __init__(self, id, node_id, d_x_restrained, d_y_restrained, r_z_restrained):
        """
        Método inicializador.

        Parâmetros:
            id (int): identificador.
            node_id (int): identificador do nó que contém o apoio.
            d_x_restrained (int): 1 se a translação em x_global é restringida, 0 se não.
            d_y_restrained (int): 1 se a direção y_global é restringida, 0 se não.
            r_z_restrained (int): 1 se a rotação em torno de z é restringida, 0 se não.
        """
        self.id = id
        self.node_id = node_id
        self.restrained_coords = [d_x_restrained, d_y_restrained, r_z_restrained]


class Node:
    """Classe que define um nó do pórtico plano."""

    def __init__(self, id, x, y):
        """
        Método inicializador.

        Parâmetros:
            id (int): identificador.
            x (float): coordenada x do nó.
            y (float): coordenada y do nó.
        """
        self.id = id
        self.x = x
        self.y = y
        self.coord_numbers = [self.id * 3, self.id * 3 + 1, self.id * 3 + 2]
        self.has_support = False
        self.has_applied_node_load = False

    def set_support(self, support: Support):
        """
        Método que aplica apoio em um nó.

        Parâmetros:
            support (Support): objeto da classe Support com informações do apoio.
        """
        self.has_support = True
        self.restrained_coords = support.restrained_coords
        self.support_reactions = np.zeros([3])

    def set_load(self, applied_node_load: NodeLoad):
        """
        Método que aplica carga em um nó.

        Parâmetros:
            applied_node_load (NodeLoad): objeto da classe NodeLoad com informações da carga.
        """
        self.has_applied_node_load = True
        self.applied_node_load = applied_node_load


class Member:
    """Classe que define uma barra do pórtico."""

    def __init__(self, id, first_node: Node, second_node: Node, E, A, I):
        """
        Método inicializador.

        Parâmetros:
            id (int): identificador.
            first_node (Node): objeto da classe Node com informações do primeiro nó da barra.
            second_node (Node): objeto da classe Node com informações do segundo nó da barra.
            E (float): módulo de elasticidade longitudinal.
            A (float): área da seção transversal.
            I (float): momento de inércia da seção transversal.
        """
        self.id = id
        self.first_node = first_node
        self.second_node = second_node
        self.E = E
        self.A = A
        self.I = I
        self.x1 = self.first_node.x
        self.y1 = self.first_node.y
        self.x2 = self.second_node.x
        self.y2 = self.second_node.y
        self.L = np.sqrt((self.x1 - self.x2) ** 2 + (self.y1 - self.y2) ** 2)
        self.cs = (self.x2 - self.x1) / self.L
        self.sn = (self.y2 - self.y1) / self.L
        self.has_applied_member_load = False

        # Vetor com as reações para elemento engastado-engastado
        self.fixed_end_forces_local = np.zeros([6])
        self.fixed_end_forces_global = np.zeros([6])

        self.applied_member_loads = []
        self.calculate_k()
        self.calculate_T()
        self.calculate_K()

    def set_applied_member_load(self, member_load: MemberLoad):
        """
        Método que aplica carregamento distribuído em uma barra.

        Parâmetros:
            member_load (MemberLoad): objeto da classe MemberLoad com informações do carregamento.
        """
        self.has_applied_member_load = True
        self.applied_member_loads.append(member_load)
        self.calculate_fixed_end_forces(member_load.id)

    def calculate_k(self):
        """Método que calcula matriz de rigidez da barra no SCL."""
        E, A, I, L = self.E, self.A, self.I, self.L
        k = np.array(
            [
                [E * A / L, 0, 0, -E * A / L, 0, 0],
                [
                    0,
                    12 * E * I / L ** 3,
                    6 * E * I / L ** 2,
                    0,
                    -12 * E * I / L ** 3,
                    6 * E * I / L ** 2,
                ],
                [
                    0,
                    6 * E * I / L ** 2,
                    4 * E * I / L,
                    0,
                    -6 * E * I / L ** 2,
                    2 * E * I / L,
                ],
                [-E * A / L, 0, 0, E * A / L, 0, 0],
                [
                    0,
                    -12 * E * I / L ** 3,
                    -6 * E * I / L ** 2,
                    0,
                    12 * E * I / L ** 3,
                    -6 * E * I / L ** 2,
                ],
                [
                    0,
                    6 * E * I / L ** 2,
                    2 * E * I / L,
                    0,
                    -6 * E * I / L ** 2,
                    4 * E * I / L,
                ],
            ]
        )

        self.k = k

    def calculate_T(self):
        """Método que calcula a matriz de rotação."""
        cs, sn = self.cs, self.sn
        T = np.array(
            [
                [cs, sn, 0, 0, 0, 0],
                [-sn, cs, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, cs, sn, 0],
                [0, 0, 0, -sn, cs, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        self.T = T

    def calculate_K(self):
        """Método que calcula matriz de rigidez da barra no SCG (triplo produto matricial)."""
        k = self.k
        T = self.T
        K = np.dot(np.dot(T.T, k), T)
        self.K = K

    def calculate_fixed_end_forces(self, load_id):
        """
        Método que calcula vetor de forças nas extremidades da barra devido aos
        carregamentos distribuídos no SCL e SCG.

        Parâmetros:
            load_id: identificador do carregamento.
        """
        load = self.applied_member_loads[load_id]
        # Reações de barra formada pelos nós j e k
        # A = axial, S = cisalhamento, M = momento
        FAj = FSj = FMj = FAk = FSk = FMk = 0
        L = self.L

        if self.has_applied_member_load:
            qx, qy = load.qx, load.qy
            if "UNIFORM" in load.load_type:
                FAj = -qx * L / 2
                FSj = -qy * L / 2
                FMj = -qy * L ** 2 / 12
                FAk = -qx * L / 2
                FSk = -qy * L / 2
                FMk = +qy * L ** 2 / 12

            self.fixed_end_forces_local[0] += FAj
            self.fixed_end_forces_local[1] += FSj
            self.fixed_end_forces_local[2] += FMj
            self.fixed_end_forces_local[3] += FAk
            self.fixed_end_forces_local[4] += FSk
            self.fixed_end_forces_local[5] += FMk

            self.fixed_end_forces_global += np.dot(
                self.T.T, self.fixed_end_forces_local
            )

    def set_member_end_displacements(self, u, v):
        """
        Método que armazena os deslocamentos globais das extremidades da barra
        no SCL e SCG.

        Parâmetros:
            u (np.array): deslocamentos no SCL
            v (np.array): deslocamentos no SCG
        """
        self.u = u
        self.v = v

    def set_member_end_forces(self, Q, F):
        """
        Método que armazena os esforços das extremidades da barra
        no SCL e SCG.

        Parâmetros:
            Q (np.array): esforços no SCL
            F (np.array): esforços no SCG
        """
        self.Q = Q
        self.F = F


class Diagrams:
    """Classe que calcula os valores dos diagramas de esforços internos solicitantes."""

    def __init__(self, member: Member, step_size=20):
        """
        Método inicializador.

        Parâmetros:
            member (Member): barra do pórtico
            step_size (int): quanto maior, mais pontos o diagrama terá.
        """
        self.member = member
        self.step_size = step_size
        self.x = np.arange(
            0,
            self.member.L + self.member.L / self.step_size,
            self.member.L / self.step_size,
        )
        self.N = self.calculate_axial_force_vector()
        self.V = self.calculate_shear_force_vector()
        self.M = self.calculate_bending_moment_vector()

    def calculate_axial_force_vector(self):
        """Método que calcula esforço normal."""
        axial_force = -self.member.Q[0] + 0 * self.x
        for load in self.member.applied_member_loads:
            if "UNIFORM" in load.load_type:
                axial_force -= load.qx * self.x
        return axial_force

    def calculate_shear_force_vector(self):
        """Método que calcula esforço cortante."""
        shear_force = self.member.Q[1] + 0 * self.x
        for load in self.member.applied_member_loads:
            if "UNIFORM" in load.load_type:
                shear_force += load.qy * self.x
        return shear_force

    def calculate_bending_moment_vector(self):
        """Método que calcula momento fletor."""
        bending_moment = -self.member.Q[2] + self.member.Q[1] * self.x
        for load in self.member.applied_member_loads:
            if "UNIFORM" in load.load_type:
                bending_moment += load.qy * self.x ** 2 / 2
        return bending_moment

    def show_force_diagram(self, type_force, scale_factor=0.05, rotate=True):
        """
        Método que plota os diagramas de esforços internos solicitantes.

        Parâmetros:
            type_force (str):
                "N" = Esforço normal
                "V" = Esforço cortante
                "M" = Momento fletor
            scale_factor (float): fator de escala para exibição dos diagramas
            rotate (bool): apresenta os esforços em todo o pórtico se verdadeiro
        """
        if type_force == "N":
            f = self.N * scale_factor
        elif type_force == "V":
            f = self.V * scale_factor
        elif type_force == "M":
            f = -self.M * scale_factor
        if rotate:
            cs = self.member.cs
            sn = self.member.sn
            node_1 = self.member.first_node
            node_2 = self.member.second_node
            plt.plot([node_1.x, node_2.x], [node_1.y, node_2.y], "k")
            plt.plot(
                self.x * cs + f * -sn + node_1.x, self.x * sn + f * cs + node_1.y, "b"
            )
            plt.plot(
                [
                    self.x[0] * cs + self.x[0] * -sn + node_1.x,
                    self.x[0] * cs + f[0] * -sn + node_1.x,
                ],
                [
                    self.x[0] * sn + self.x[0] * cs + node_1.y,
                    self.x[0] * sn + f[0] * cs + node_1.y,
                ],
                "b",
            )
            plt.plot(
                [
                    self.x[-1] * cs + self.x[0] * -sn + node_1.x,
                    self.x[-1] * cs + f[-1] * -sn + node_1.x,
                ],
                [
                    self.x[-1] * sn + self.x[0] * cs + node_1.y,
                    self.x[-1] * sn + f[-1] * cs + node_1.y,
                ],
                "b",
            )
        else:
            plt.plot(self.x, np.zeros(len(self.x)), "k")
            plt.plot(self.x, f, "b")
            plt.plot([self.x[0], self.x[0]], [self.x[0], f[0]], "b")
            plt.plot([self.x[-1], self.x[-1]], [self.x[0], f[-1]], "b")


class Model:
    """Classe que define um modelo de pórtico."""

    def __init__(self, input_file_path):
        """
        Método inicializador

        Parâmetros:
            input_file_path (str): caminho do arquivo de entrada.
        """
        self.input_file_path = input_file_path
        self.set_model()

    def read_input_file(self):
        """Método que lê o conteúdo do arquivo de entrada."""
        with open(self.input_file_path, "r", encoding="utf-8") as f:
            self.raw_data = f.read().split("\n")

    def set_nodes(self):
        """Método que faz a leitura dos dados referentes aos nós."""
        node_header_index = self.raw_data.index("% NODES %")
        num_nodes_index = node_header_index + 1
        first_node_index = num_nodes_index + 1

        self.num_nodes = int(self.raw_data[num_nodes_index])
        self.nodes = [None] * self.num_nodes

        node_data = self.raw_data[first_node_index : first_node_index + self.num_nodes]

        for i, node in enumerate(node_data):
            x, y = node.split()
            self.nodes[i] = Node(i, float(x), float(y))

    def set_supports(self):
        """Método que faz a leitura dos dados referentes aos apoios."""
        support_header_index = self.raw_data.index("% SUPPORTS %")
        num_supports_index = support_header_index + 1
        first_support_index = num_supports_index + 1

        self.num_supports = int(self.raw_data[num_supports_index])
        self.supports = [None] * self.num_supports

        support_data = self.raw_data[
            first_support_index : first_support_index + self.num_supports
        ]

        for i, support in enumerate(support_data):
            node_id, d_x, d_y, r_z = support.split()
            for node in self.nodes:
                if node.id == int(node_id) - 1:
                    self.supports[i] = Support(i, node.id, int(d_x), int(d_y), int(r_z))
                    node.set_support(self.supports[i])

    def set_materials(self):
        """Método que faz a leitura dos dados referentes aos materiais."""
        material_header_index = self.raw_data.index("% MATERIALS %")
        num_materials_index = material_header_index + 1
        first_material_index = num_materials_index + 1

        self.num_materials = int(self.raw_data[num_materials_index])
        self.materials = [None] * self.num_materials

        material_data = self.raw_data[
            first_material_index : first_material_index + self.num_materials
        ]

        for i, young_modulus in enumerate(material_data):
            self.materials[i] = Material(i, float(young_modulus))

    def set_cross_sections(self):
        """Método que faz a leitura dos dados referentes às seções transversais."""
        cross_section_header_index = self.raw_data.index("% CROSS-SECTIONS %")
        num_cross_sections_index = cross_section_header_index + 1
        first_cross_section_index = num_cross_sections_index + 1

        self.num_cross_sections = int(self.raw_data[num_cross_sections_index])
        self.cross_sections = [None] * self.num_cross_sections

        cross_section_data = self.raw_data[
            first_cross_section_index : first_cross_section_index
            + self.num_cross_sections
        ]

        for i, cross_section in enumerate(cross_section_data):
            area, moment_of_inertia = cross_section.split()
            self.cross_sections[i] = CrossSection(
                i, float(area), float(moment_of_inertia)
            )

    def set_members(self):
        """Método que faz a leitura dos dados referentes às barras."""
        member_header_index = self.raw_data.index("% MEMBERS %")
        num_members_index = member_header_index + 1
        first_member_index = num_members_index + 1

        self.num_members = int(self.raw_data[num_members_index])
        self.members = [None] * self.num_members

        member_data = self.raw_data[
            first_member_index : first_member_index + self.num_members
        ]

        for i, member in enumerate(member_data):
            first_node, second_node, material_type, cross_section_type = member.split()
            material = self.materials[int(material_type) - 1]
            cross_section = self.cross_sections[int(cross_section_type) - 1]
            self.members[i] = Member(
                i,
                self.nodes[int(first_node) - 1],
                self.nodes[int(second_node) - 1],
                material.E,
                cross_section.A,
                cross_section.I,
            )

    def set_applied_node_loads(self):
        """Método que faz a leitura das informações referentes às cargas nodais."""
        node_load_header_index = self.raw_data.index("% NODE LOADS %")
        num_node_loads_index = node_load_header_index + 1
        first_node_load_index = num_node_loads_index + 1

        self.num_node_loads = int(self.raw_data[num_node_loads_index])
        self.node_loads = [None] * self.num_node_loads

        node_load_data = self.raw_data[
            first_node_load_index : first_node_load_index + self.num_node_loads
        ]

        for i, load in enumerate(node_load_data):
            node_id, Fx, Fy, Mz = load.split()
            for node in self.nodes:
                if node.id == (int(node_id) - 1):
                    node_load = NodeLoad(i, node.id, float(Fx), float(Fy), float(Mz))
                    node.set_load(node_load)
                    self.node_loads.append(node_load)

    def set_applied_member_loads(self):
        """Método que faz a leitura das informações referentes aos carregamentos distribuídos."""
        member_load_header_index = self.raw_data.index("% MEMBER LOADS %")
        num_member_loads_index = member_load_header_index + 1
        first_member_load_index = num_member_loads_index + 1

        self.num_member_loads = int(self.raw_data[num_member_loads_index])
        self.member_loads = [None] * self.num_member_loads

        member_load_data = self.raw_data[
            first_member_load_index : first_member_load_index + self.num_member_loads
        ]

        for i, load in enumerate(member_load_data):
            member_id, load_type, *load_data = load.split()
            if "UNIFORM" in load_type:
                if load_type == "UNIFORM-GLOBAL":
                    qx_global, qy_global = float(load_data[0]), float(load_data[1])
                    cs = self.members[int(member_id) - 1].cs
                    sn = self.members[int(member_id) - 1].sn
                    qx_local = qx_global * cs + qy_global * sn
                    qy_local = -qx_global * sn + qy_global * cs
                elif load_type == "UNIFORM-LOCAL":
                    qx_local, qy_local = float(load_data[0]), float(load_data[1])
                member_load = MemberLoad(i, load_type, qx_local, qy_local)
                for member in self.members:
                    if member.id == (int(member_id) - 1):
                        member.set_applied_member_load(member_load)
                        self.member_loads.append(member_load)

    def set_model(self):
        """Método que faz a leitura de todas as informações necessárias."""
        self.read_input_file()
        self.set_nodes()
        self.set_supports()
        self.set_materials()
        self.set_cross_sections()
        self.set_members()
        self.set_applied_node_loads()
        self.set_applied_member_loads()

    def show_all_node_informations(self):
        """Método que mostra na tela todas as informações dos nós do pórtico."""
        for node in self.nodes:
            print(vars(node))

    def show_all_member_informations(self):
        """Método que mostra na tela todas as informações das barras do pórtico."""
        for member in self.members:
            print(vars(member))

    def showModelInformations(self):
        """Método que mostra na tela todas as informações do modelo."""
        print(vars(self))


class Frame_2D:
    """Classe que define e calcula um pórtico plano."""

    def __init__(self, input_file_path):
        """
        Método inicializador.

        Parâmetros:
            input_file_path (str): caminho do arquivo de entrada.
        """
        self.model = Model(input_file_path)
        self.calculate_S()
        self.calculate_global_node_load_vector()
        self.calculate_global_equivalent_node_load_vector()

    def calculate_S(self):
        """Método que calcula a matriz de rigidez do pórtico"""
        self.S = np.zeros([3 * self.model.num_nodes, 3 * self.model.num_nodes])
        for member in self.model.members:
            self.assembly_S(member)

    def assembly_S(self, member: Member):
        """
        Método que faz o mapeamento da matriz de rigidez da barra no SCG para a
        matriz de rigidez global do pórtico.

        Parâmetros:
            member (Member): objeto da classe Member que representa a barra.
        """
        for row_K in range(6):
            # Verifica se a linha está associado ao primeiro nó da barra
            if row_K < 3:
                row_S = member.first_node.coord_numbers[row_K]
            else:
                row_S = member.second_node.coord_numbers[row_K - 3]
            for column_K in range(6):
                # Verifica se a linha está associado ao segundo nó da barra
                if column_K < 3:
                    column_S = member.first_node.coord_numbers[column_K]
                else:
                    column_S = member.second_node.coord_numbers[column_K - 3]
                self.S[row_S, column_S] += member.K[row_K, column_K]

    def calculate_global_node_load_vector(self):
        """Método que calcula o vetor de forças nodais."""
        self.global_node_load_vector = np.zeros([3 * self.model.num_nodes])

        for node in self.model.nodes:
            if node.has_applied_node_load:
                Fx, Fy, Mz = node.applied_node_load.vector
                for coord_number, force in zip(node.coord_numbers, (Fx, Fy, Mz)):
                    self.global_node_load_vector[coord_number] += force

    def calculate_global_equivalent_node_load_vector(self):
        """Método que calcula o vetor de forças nodais equivalentes."""
        self.global_equivalent_node_load_vector = np.zeros(3 * self.model.num_nodes)

        for member in self.model.members:
            if member.has_applied_member_load:
                for i in range(6):
                    if i < 3:
                        j = member.first_node.coord_numbers[i]
                    else:
                        j = member.second_node.coord_numbers[i - 3]
                    self.global_equivalent_node_load_vector[
                        j
                    ] += member.fixed_end_forces_global[i]
        # Inversão dos valores
        self.global_equivalent_node_load_vector *= -1

    def calculate_global_displacements(self):
        """Método que calcula deslocamentos globais do pórtico."""

        # Matrizes e vetores auxiliares
        aux_S = np.copy(self.S)
        aux_P = np.copy(self.global_node_load_vector)
        aux_Pe = np.copy(self.global_equivalent_node_load_vector)

        # Implementação do método pênalti
        for node in self.model.nodes:
            if node.has_support:
                for coord_number, restrained_coord in zip(
                    node.coord_numbers, node.restrained_coords
                ):
                    if restrained_coord == 1:
                        i = coord_number
                        for row_S in range(3 * self.model.num_nodes):
                            for column_S in range(3 * self.model.num_nodes):
                                if row_S == i or column_S == i:
                                    aux_S[row_S, column_S] = 0
                        aux_S[i, i] = 1
                        aux_P[i] = 0
                        aux_Pe[i] = 0
        self.global_displacements = np.linalg.solve(aux_S, aux_P + aux_Pe)

    def calculate_member_displacements(self):
        """Método que calcula deslocamentos das extremidades das barras no SCG e SCL."""
        self.v = np.zeros([self.model.num_members, 6])
        self.u = np.zeros([self.model.num_members, 6])

        # Caso precise rodar esse método isoladamente, descomente a linha abaixo
        # self.calculate_global_displacements()

        for member in self.model.members:
            for i, coord in zip(range(3), member.first_node.coord_numbers):
                self.v[member.id, i] = self.global_displacements[coord]
            for i, coord in zip(range(3, 6), member.second_node.coord_numbers):
                self.v[member.id, i] = self.global_displacements[coord]
            self.u[member.id] = np.dot(member.T, self.v[member.id])
            member.set_member_end_displacements(self.u[member.id], self.v[member.id])

    def calculate_member_forces(self):
        """Método que calcula esforços das extremidades das barras no SCG e SCL."""
        self.Q = np.zeros([self.model.num_members, 6])
        self.F = np.zeros([self.model.num_members, 6])

        # Caso precise rodar esse método isoladamente, descomente a linha abaixo
        # self.calculate_member_displacements()

        for member in self.model.members:
            self.Q[member.id] = (
                np.dot(member.k, self.u[member.id]) + member.fixed_end_forces_local
            )
            self.F[member.id] = np.dot(member.T.T, self.Q[member.id])
            member.set_member_end_forces(self.Q[member.id], self.F[member.id])

    def calculate_support_reactions(self):
        """Método que calcula reações de apoio do pórtico."""
        self.support_reactions = np.zeros([self.model.num_supports, 3])

        # Caso precise rodar esse método isoladamente, descomente a linha abaixo
        # self.calculate_member_forces()

        for support in self.model.supports:
            reaction = np.zeros(3)
            for member in self.model.members:
                if member.first_node.id == support.node_id:
                    for i in range(3):
                        if member.first_node.restrained_coords[i]:
                            reaction[i] += member.F[i]
                    member.first_node.support_reactions = reaction
                    self.support_reactions[support.id] = reaction
                if member.second_node.id == support.node_id:
                    for i in range(3):
                        if member.second_node.restrained_coords[i]:
                            reaction[i] += member.F[i + 3]
                    member.second_node.support_reactions = reaction
                    self.support_reactions[support.id] = reaction

    def solve_frame(self):
        """Método que resolve o pórtico."""
        self.calculate_global_displacements()
        self.calculate_member_displacements()
        self.calculate_member_forces()
        self.calculate_support_reactions()


class Results:
    """Classe que gera e apresenta os resultados da análise."""

    def __init__(self, frame: Frame_2D):
        """
        Método inicializador.

        Parâmetros:
            frame (Frame_2D): objeto da classe Frame_2D com informações do pórtico.
        """
        self.frame = frame
        self.model = frame.model
        self.input_file_path = Path(self.model.input_file_path)
        self.output_file_path = (
            self.input_file_path.parent / f"Resultados - {self.input_file_path.name}"
        )
        self.template_path = Path(__file__).parent / "template/template_results.txt"
        with open(self.template_path, "r", encoding="utf-8") as f:
            self.template = f.read().split("\n")
        self.output = self.template.copy()

    def write_header(self):
        """Método que escreve cabeçalho dos resultados."""
        case_index = self.output.index("Caso:")
        num_nodes_index = self.output.index("Número de nós:")
        num_members_index = self.output.index("Número de barras:")

        self.output[case_index] += f" {self.input_file_path.stem}"
        self.output[num_nodes_index] += f" {self.model.num_nodes}"
        self.output[num_members_index] += f" {self.model.num_members}"

    def write_nodes(self):
        """Método que escreve informações dos nós."""
        row_to_write = self.output.index("Nós:") + 2

        for i, node in enumerate(self.model.nodes):
            self.output.insert(
                row_to_write, f"{i + 1} | {node.x:+6.4E} | {node.y:+6.4E}"
            )
            row_to_write += 1

    def write_members(self):
        """Método que escreve informações das barras."""
        row_to_write = self.output.index("Barras:") + 2

        for i, member in enumerate(self.model.members):
            self.output.insert(
                row_to_write,
                f"{i + 1} | {member.first_node.id + 1} | {member.second_node.id + 1} | {member.E:6.4E} | {member.A:6.4E} | {member.I:6.4E}",
            )
            row_to_write += 1

    def write_supports(self):
        """Método que escreve informações dos apoios."""
        row_to_write = self.output.index("Apoios:") + 2

        for support in self.model.supports:
            node_id = support.node_id
            dx = "Sim" if support.restrained_coords[0] == 1 else "Não"
            dy = "Sim" if support.restrained_coords[1] == 1 else "Não"
            rz = "Sim" if support.restrained_coords[2] == 1 else "Não"
            self.output.insert(row_to_write, f"{node_id + 1} | {dx} | {dy} | {rz}")
            row_to_write += 1

    def write_applied_node_loads(self):
        """Método que escreve informações das cargas nodais aplicadas."""
        row_to_write = self.output.index("Forças nodais:") + 2

        for node in self.model.nodes:
            if node.has_applied_node_load:
                Fx, Fy, Mz = node.applied_node_load.vector

                self.output.insert(
                    row_to_write,
                    f"{node.id + 1} | {Fx:+6.4E} | {Fy:+6.4E} | {Mz:+6.4E}",
                )
                row_to_write += 1

    def write_applied_member_loads(self):
        """Método que escreve informações das cargas distribuídas aplicadas."""
        row_to_write = self.output.index("Cargas distribuídas:") + 2

        for member in self.model.members:
            if member.has_applied_member_load:
                for load in member.applied_member_loads:
                    if "UNIFORM" in load.load_type:
                        if load.load_type == "UNIFORM-GLOBAL":
                            qx_global = load.qx * member.cs - load.qy * member.sn
                            qy_global = load.qx * member.sn + load.qy * member.cs
                            self.output.insert(
                                row_to_write,
                                f"{member.id + 1} | UNIFORME-GLOBAL | {qx_global:+6.4E} | {qy_global:+6.4E}",
                            )
                        elif load.load_type == "UNIFORM-LOCAL":
                            self.output.insert(
                                row_to_write,
                                f"{member.id + 1} | UNIFORME-LOCAL | {load.qx:+6.4E} | {load.qy:+6.4E}",
                            )
                        row_to_write += 1

    def write_support_reactions(self):
        """Método que escreve informações das reações de apoio."""
        row_to_write = self.output.index("Reações de apoio:") + 2

        for support in self.model.supports:
            FRx, FRy, MRz = self.frame.support_reactions[support.id]
            node_id = support.node_id
            self.output.insert(
                row_to_write,
                f"{node_id + 1} | {FRx:+6.4E} | {FRy:+6.4E} | {MRz:+6.4E}",
            )
            row_to_write += 1

    def write_member_end_forces(self):
        """Método que escreve informações dos esforços internos solicitantes."""
        row_to_write = self.output.index("Esforços internos solicitantes:") + 2

        for member in self.model.members:
            N1, Q1, M1 = member.Q[:3]
            N2, Q2, M2 = member.Q[3:]
            self.output.insert(
                row_to_write,
                f"{member.id + 1} | {member.first_node.id + 1} | {-N1:+6.4E} | {Q1:+6.4E} | {-M1:+6.4E}",
            )
            row_to_write += 1
            self.output.insert(
                row_to_write,
                f"{member.id + 1} | {member.second_node.id + 1} | {N2:+6.4E} | {-Q2:+6.4E} | {M2:+6.4E}",
            )
            row_to_write += 1
            self.output.insert(row_to_write, "---")
            row_to_write += 1

    def write_node_global_displacements(self):
        """Método que escreve informações dos deslocamentos nodais no SCG."""
        row_to_write = self.output.index("Deslocamentos nodais:") + 2

        for node in self.model.nodes:
            Dx, Dy, Rz = self.frame.global_displacements[
                node.coord_numbers[0] : node.coord_numbers[2] + 1
            ]
            self.output.insert(
                row_to_write,
                f"{node.id + 1} | {Dx:+6.4E} | {Dy:+6.4E} | {Rz:+6.4E}",
            )
            row_to_write += 1

    def write_results(self, save_file=False):
        """
        Método que escreve todos os resultados.

        Parâmetros:
            save_file (bool): salva arquivo com resultados se True.
        """
        self.write_header()
        self.write_nodes()
        self.write_members()
        self.write_supports()
        self.write_applied_node_loads()
        self.write_applied_member_loads()
        self.write_support_reactions()
        self.write_member_end_forces()
        self.write_node_global_displacements()
        print("\n".join(self.output))
        if save_file:
            with open(self.output_file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(self.output))

    def show_axial_force_diagram(self, scale_factor=0.05):
        """
        Método que apresenta o diagrama de esforço normal.

        Parâmetros:
            scale_factor (float): fator de escala para exibição do diagrama
        """
        plt.axis("equal")
        plt.axis("off")
        for member in self.model.members:
            diagram = Diagrams(member)
            diagram.show_force_diagram("N", scale_factor)
        plt.show()

    def show_shear_force_diagram(self, scale_factor=0.05):
        """
        Método que apresenta o diagrama de esforço cortante.

        Parâmetros:
            scale_factor (float): fator de escala para exibição do diagrama
        """
        plt.axis("equal")
        plt.axis("off")
        for member in self.model.members:
            diagram = Diagrams(member)
            diagram.show_force_diagram("V", scale_factor)
        plt.show()

    def show_bending_moment_diagram(self, scale_factor=0.05):
        """
        Método que apresenta o diagrama de momento fletor.

        Parâmetros:
            scale_factor (float): fator de escala para exibição do diagrama
        """
        plt.axis("equal")
        plt.axis("off")
        for member in self.model.members:
            diagram = Diagrams(member)
            diagram.show_force_diagram("M", scale_factor)
        plt.show()


if __name__ == "__main__":
    # Pórtico 1
    portico_1 = Frame_2D(Path(__file__).parent / "data" / "Exemplo 1 - Pórtico.txt")
    portico_1.solve_frame()
    results_1 = Results(portico_1)
    results_1.write_results(save_file=True)
    results_1.show_axial_force_diagram()
    results_1.show_shear_force_diagram()
    results_1.show_bending_moment_diagram()

    # Pórtico 2
    portico_2 = Frame_2D(Path(__file__).parent / "data" / "Exemplo 2 - Pórtico.txt")
    portico_2.solve_frame()
    results_2 = Results(portico_2)
    results_2.write_results(save_file=True)
    results_2.show_axial_force_diagram()
    results_2.show_shear_force_diagram()
    results_2.show_bending_moment_diagram()

    # Pórtico 3
    portico_3 = Frame_2D(Path(__file__).parent / "data" / "Exemplo 3 - Pórtico.txt")
    portico_3.solve_frame()
    results_3 = Results(portico_3)
    results_3.write_results(save_file=True)
    results_3.show_axial_force_diagram()
    results_3.show_shear_force_diagram()
    results_3.show_bending_moment_diagram()

    # Pórtico 4
    portico_4 = Frame_2D(Path(__file__).parent / "data" / "Exemplo 4 - Pórtico.txt")
    portico_4.solve_frame()
    results_4 = Results(portico_4)
    results_4.write_results(save_file=True)
    results_4.show_axial_force_diagram()
    results_4.show_shear_force_diagram(scale_factor=0.01)
    results_4.show_bending_moment_diagram(scale_factor=0.005)
