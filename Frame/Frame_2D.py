import numpy as np


class Material:
    def __init__(self, id, E):
        self.id = id
        self.E = E


class CrossSection:
    def __init__(self, id, A, I):
        self.id = id
        self.A = A
        self.I = I


class NodeLoad:
    def __init__(self, id, Fx, Fy, Mz):
        self.id = id
        self.Fx = Fx
        self.Fy = Fy
        self.Mz = Mz
        self.vector = np.array([Fx, Fy, Mz])


class MemberLoad:
    def __init__(self, id, load_type, *load_data):
        self.id = id
        self.load_type = load_type
        if self.load_type in ("UNIFORM", "UNIFORM-H"):
            self.w1 = self.w2 = load_data[0]
        elif self.load_type in ("TRIANGULAR", "TRAPEZOIDAL"):
            self.w1 = load_data[0]
            self.w2 = load_data[1]


class Node:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y
        self.coord_numbers = [self.id * 3, self.id * 3 + 1, self.id * 3 + 2]
        self.has_support = False
        self.has_applied_node_load = False

    def setSupport(self, d_x_restrained, d_y_restrained, r_z_restrained):
        self.has_support = True
        self.restrained_coords = [d_x_restrained, d_y_restrained, r_z_restrained]

    def setLoad(self, applied_node_load: NodeLoad):
        self.has_applied_node_load = True
        self.applied_node_load = applied_node_load


class Member:
    def __init__(self, id, first_node: Node, second_node: Node, E, A, I):
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
        self.fixed_end_forces_local = np.zeros([6])
        self.fixed_end_forces_global = np.zeros([6])
        self.applied_member_loads = []
        self.calculate_k()
        self.calculate_T()
        self.calculate_K()

    def setAppliedMemberLoad(self, member_load: MemberLoad):
        self.has_applied_member_load = True
        self.applied_member_loads.append(member_load)
        self.calculate_fixed_end_forces(member_load.id)

    def calculate_k(self):
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
        k = self.k
        T = self.T
        K = np.dot(np.dot(T.T, k), T)
        self.K = K

    def calculate_fixed_end_forces(self, load_id):
        load = self.applied_member_loads[load_id]
        FAb = FSb = FMb = FAe = FSe = FMe = 0
        L, cs, sn = (self.L, self.cs, self.sn)

        if self.has_applied_member_load:
            w1, w2 = load.w1, load.w2
            if load.load_type == "UNIFORM-H":
                FAb = w1 * L / 2
                FAe = w1 * L / 2
            else:
                FSb = FSe = (7 * w1 * L + 3 * w2 * L) / 20
                FMb = w1 * L ** 2 / 20 + w2 * L ** 2 / 30
                FMe = -w1 * L ** 2 / 30 - w2 * L ** 2 / 20

            self.fixed_end_forces_local[0] += -FAb
            self.fixed_end_forces_local[1] += -FSb
            self.fixed_end_forces_local[2] += -FMb
            self.fixed_end_forces_local[3] += -FAe
            self.fixed_end_forces_local[4] += -FSe
            self.fixed_end_forces_local[5] += -FMe

            self.fixed_end_forces_global += np.dot(
                self.T.T, self.fixed_end_forces_local
            )

    def setMemberDisplacements(self, u, v):
        self.u = u
        self.v = v

    def setMemberEndForces(self, Q, F):
        self.Q = Q
        self.F = F


class Model:
    def __init__(self, input_file_path):
        self.input_file_path = input_file_path
        self.setModel()

    def read_input_file(self):
        with open(self.input_file_path, "r", encoding="utf-8") as f:
            raw_data = f.read().split("\n")
        return raw_data

    def setModel(self):
        raw_data = self.read_input_file()

        # Leitura dos dados referentes aos nós
        node_header_index = raw_data.index("% NODES %")
        num_nodes_index = node_header_index + 1
        first_node_index = num_nodes_index + 1

        self.num_nodes = int(raw_data[num_nodes_index])
        self.nodes = [None] * self.num_nodes

        node_data = raw_data[first_node_index : first_node_index + self.num_nodes]

        for i, node in enumerate(node_data):
            x, y = node.split()
            self.nodes[i] = Node(i, float(x), float(y))

        # Leitura dos dados referentes aos apoios
        support_header_index = raw_data.index("% SUPPORTS %")
        num_supports_index = support_header_index + 1
        first_support_index = num_supports_index + 1

        self.num_supports = int(raw_data[num_supports_index])

        support_data = raw_data[
            first_support_index : first_support_index + self.num_supports
        ]

        for support in support_data:
            node_id, d_x, d_y, r_z = support.split()
            for node in self.nodes:
                if node.id == int(node_id) - 1:
                    node.setSupport(int(d_x), int(d_y), int(r_z))

        # Leitura dos dados referentes aos materiais
        material_header_index = raw_data.index("% MATERIALS %")
        num_materials_index = material_header_index + 1
        first_material_index = num_materials_index + 1

        self.num_materials = int(raw_data[num_materials_index])
        self.materials = [None] * self.num_materials

        material_data = raw_data[
            first_material_index : first_material_index + self.num_materials
        ]

        for i, young_modulus in enumerate(material_data):
            self.materials[i] = Material(i, float(young_modulus))

        # Leitura dos dados referentes às seções transversais
        cross_section_header_index = raw_data.index("% CROSS-SECTIONS %")
        num_cross_sections_index = cross_section_header_index + 1
        first_cross_section_index = num_cross_sections_index + 1

        self.num_cross_sections = int(raw_data[num_cross_sections_index])
        self.cross_sections = [None] * self.num_cross_sections

        cross_section_data = raw_data[
            first_cross_section_index : first_cross_section_index
            + self.num_cross_sections
        ]

        for i, cross_section in enumerate(cross_section_data):
            area, moment_of_inertia = cross_section.split()
            self.cross_sections[i] = CrossSection(
                i, float(area), float(moment_of_inertia)
            )

        # Leitura dos dados referentes aos elementos
        member_header_index = raw_data.index("% MEMBERS %")
        num_members_index = member_header_index + 1
        first_member_index = num_members_index + 1

        self.num_members = int(raw_data[num_members_index])
        self.members = [None] * self.num_members

        member_data = raw_data[
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

        # Leitura das informações referentes às cargas nodais
        node_load_header_index = raw_data.index("% NODE LOADS %")
        num_node_loads_index = node_load_header_index + 1
        first_node_load_index = num_node_loads_index + 1

        self.num_node_loads = int(raw_data[num_node_loads_index])

        node_load_data = raw_data[
            first_node_load_index : first_node_load_index + self.num_node_loads
        ]

        for i, load in enumerate(node_load_data):
            node_id, Fx, Fy, Mz = load.split()
            node_load = NodeLoad(i, float(Fx), float(Fy), float(Mz))
            for node in self.nodes:
                if node.id == (int(node_id) - 1):
                    node.setLoad(node_load)

        # Leitura das informações referentes aos carregamentos distribuídos
        member_load_header_index = raw_data.index("% MEMBER LOADS %")
        num_member_loads_index = member_load_header_index + 1
        first_member_load_index = num_member_loads_index + 1

        self.num_member_loads = int(raw_data[num_member_loads_index])

        member_load_data = raw_data[
            first_member_load_index : first_member_load_index + self.num_member_loads
        ]

        for i, load in enumerate(member_load_data):
            member_id, load_type, *load_data = load.split()
            if load_type in ("UNIFORM", "UNIFORM-H"):
                w = float(load_data[0])
                member_load = MemberLoad(i, load_type, w)
            elif load_type in ("TRIANGULAR", "TRAPEZOIDAL"):
                w1 = float(load_data[0])
                w2 = float(load_data[1])
                member_load = MemberLoad(i, load_type, w1, w2)
            for member in self.members:
                if member.id == (int(member_id) - 1):
                    member.setAppliedMemberLoad(member_load)

    def showAllNodeInformations(self):
        for node in self.nodes:
            print(vars(node))

    def showAllMemberInformations(self):
        for member in self.members:
            print(vars(member))

    def showModelInformations(self):
        print(vars(self))


class Frame_2D:
    def __init__(self, input_file_path):
        self.model = Model(input_file_path)
        self.calculate_S()
        self.calculate_global_node_load_vector()
        self.calculate_global_equivalent_node_load_vector()

    def calculate_S(self):
        self.S = np.zeros([3 * self.model.num_nodes, 3 * self.model.num_nodes])
        for member in self.model.members:
            self.assembly_S(member)

    def assembly_S(self, member: Member):
        for row_K in range(6):
            if row_K < 3:
                row_S = member.first_node.coord_numbers[row_K]
            else:
                row_S = member.second_node.coord_numbers[row_K - 3]
            for column_K in range(6):
                if column_K < 3:
                    column_S = member.first_node.coord_numbers[column_K]
                else:
                    column_S = member.second_node.coord_numbers[column_K - 3]
                self.S[row_S, column_S] += member.K[row_K, column_K]

    def calculate_global_node_load_vector(self):
        self.global_node_load_vector = np.zeros([3 * self.model.num_nodes])

        for node in self.model.nodes:
            if node.has_applied_node_load:
                Fx, Fy, Mz = node.applied_node_load.vector
                for coord_number, force in zip(node.coord_numbers, (Fx, Fy, Mz)):
                    self.global_node_load_vector[coord_number] += force

    def calculate_global_equivalent_node_load_vector(self):
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

    def solve_global_displacements(self):
        aux_S = np.copy(self.S)
        aux_P = np.copy(self.global_node_load_vector)
        aux_Pf = np.copy(self.global_equivalent_node_load_vector)

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
                        aux_Pf[i] = 0
        self.global_displacements = np.linalg.solve(aux_S, aux_P - aux_Pf)

    def calculate_member_displacements(self):
        self.v = np.zeros([self.model.num_members, 6])
        self.u = np.zeros([self.model.num_members, 6])

        self.solve_global_displacements()

        for member in self.model.members:
            for i, coord in zip(range(3), member.first_node.coord_numbers):
                self.v[member.id, i] = self.global_displacements[coord]
            for i, coord in zip(range(3, 6), member.second_node.coord_numbers):
                self.v[member.id, i] = self.global_displacements[coord]
            self.u[member.id] = np.dot(member.T, self.v[member.id])
            member.setMemberDisplacements(self.u[member.id], self.v[member.id])

    def calculate_member_forces(self):
        self.Q = np.zeros([self.model.num_members, 6])
        self.F = np.zeros([self.model.num_members, 6])

        self.calculate_member_displacements()

        for member in self.model.members:
            self.Q[member.id] = (
                np.dot(member.k, self.u[member.id]) + member.fixed_end_forces_local
            )
            self.F[member.id] = np.dot(member.T.T, self.Q[member.id])
            member.setMemberEndForces(self.Q[member.id], self.F[member.id])


if __name__ == "__main__":
    portico_1 = Frame_2D(
        "D:\\repositorios\\structural-matrix-analysis\\Frame\data\Exemplo 1 - Pórtico Livro.txt"
    )
    portico_1.calculate_member_forces()
