import numpy as np
from pathlib import Path


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
    def __init__(self, id, node_id, Fx, Fy, Mz):
        self.id = id
        self.node_id = node_id
        self.Fx = Fx
        self.Fy = Fy
        self.Mz = Mz
        self.vector = np.array([Fx, Fy, Mz])


class MemberLoad:
    def __init__(self, id, load_type, *load_data):
        self.id = id
        self.load_type = load_type
        if self.load_type in ("UNIFORM"):
            self.qx, self.qy = load_data


class Support:
    def __init__(self, id, node_id, d_x_restrained, d_y_restrained, r_z_restrained):
        self.id = id
        self.node_id = node_id
        self.restrained_coords = [d_x_restrained, d_y_restrained, r_z_restrained]


class Node:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y
        self.coord_numbers = [self.id * 3, self.id * 3 + 1, self.id * 3 + 2]
        self.has_support = False
        self.has_applied_node_load = False

    def set_support(self, support: Support):
        self.has_support = True
        self.restrained_coords = support.restrained_coords
        self.support_reactions = np.zeros([3])

    def set_load(self, applied_node_load: NodeLoad):
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

    def set_applied_member_load(self, member_load: MemberLoad):
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
        FAj = FSj = FMj = FAk = FSk = FMk = 0
        L = self.L

        if self.has_applied_member_load:
            qx, qy = load.qx, load.qy
            if load.load_type == "UNIFORM":
                FAj = +qx * L / 2
                FSj = +qy * L / 2
                FMj = +qy * L ** 2 / 12
                FAk = +qx * L / 2
                FSk = +qy * L / 2
                FMk = -qy * L ** 2 / 12

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
        self.u = u
        self.v = v

    def set_member_end_forces(self, Q, F):
        self.Q = Q
        self.F = F


class Diagrams:
    def __init__(self, member: Member, step_size=20):
        self.member = member
        self.x = np.arange(
            0, self.member.L + self.member.L / step_size, self.member.L / step_size
        )
        self.N = self.calculate_axial_force_vector()
        self.V = self.calculate_shear_force_vector()

    def calculate_axial_force_vector(self):
        if self.member.x1 <= self.member.x2:
            axial_force = -self.member.Q[0] + 0 * self.x
            for load in self.member.applied_member_loads:
                if load.load_type == "UNIFORM-H":
                    axial_force += load.w1 * self.x
            return axial_force
        else:
            axial_force = self.member.Q[3] + 0 * self.x
            for load in self.member.applied_member_loads:
                if load.load_type == "UNIFORM-H":
                    axial_force += -load.w1 * self.x
            return axial_force

    def calculate_shear_force_vector(self):
        if self.member.x1 <= self.member.x2:
            shear_force = self.member.Q[1] + 0 * self.x
            for load in self.member.applied_member_loads:
                if load.load_type != "UNIFORM-H":
                    shear_force += (
                        -load.w1 * self.member.L
                        + self.x * self.member.L
                        - 0.5 * self.x * load.w2
                    )
            return shear_force
        else:
            shear_force = -self.member.Q[4] + 0 * self.x
            for load in self.member.applied_member_loads:
                if load.load_type != "UNIFORM-H":
                    shear_force += -(
                        -load.w1 * self.member.L
                        + self.x * self.member.L
                        - 0.5 * self.x * load.w2
                    )
            return shear_force


class Model:
    def __init__(self, input_file_path):
        self.input_file_path = input_file_path
        self.set_model()

    def read_input_file(self):
        with open(self.input_file_path, "r", encoding="utf-8") as f:
            self.raw_data = f.read().split("\n")

    def get_nodes(self):
        # Leitura dos dados referentes aos nós
        node_header_index = self.raw_data.index("% NODES %")
        num_nodes_index = node_header_index + 1
        first_node_index = num_nodes_index + 1

        self.num_nodes = int(self.raw_data[num_nodes_index])
        self.nodes = [None] * self.num_nodes

        node_data = self.raw_data[first_node_index : first_node_index + self.num_nodes]

        for i, node in enumerate(node_data):
            x, y = node.split()
            self.nodes[i] = Node(i, float(x), float(y))

    def get_supports(self):
        # Leitura dos dados referentes aos apoios
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

    def get_materials(self):
        # Leitura dos dados referentes aos materiais
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

    def get_cross_sections(self):
        # Leitura dos dados referentes às seções transversais
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

    def get_members(self):
        # Leitura dos dados referentes aos elementos
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

    def get_applied_node_loads(self):
        # Leitura das informações referentes às cargas nodais
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

    def get_applied_member_loads(self):
        # Leitura das informações referentes aos carregamentos distribuídos
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
            if load_type in ("UNIFORM"):
                qx_global, qy_global = float(load_data[0]), float(load_data[1])
                cs = self.members[int(member_id) - 1].cs
                sn = self.members[int(member_id) - 1].sn
                qx_local = qx_global * cs + qy_global * sn
                qy_local = qx_global * sn - qy_global * cs
                member_load = MemberLoad(i, load_type, qx_local, qy_local)
                for member in self.members:
                    if member.id == (int(member_id) - 1):
                        member.set_applied_member_load(member_load)
                        self.member_loads.append(member_load)

    def set_model(self):
        self.read_input_file()
        self.get_nodes()
        self.get_supports()
        self.get_materials()
        self.get_cross_sections()
        self.get_members()
        self.get_applied_node_loads()
        self.get_applied_member_loads()

    def show_all_node_informations(self):
        for node in self.nodes:
            print(vars(node))

    def show_all_member_informations(self):
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

    def calculate_global_displacements(self):
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

        # self.calculate_global_displacements()

        for member in self.model.members:
            for i, coord in zip(range(3), member.first_node.coord_numbers):
                self.v[member.id, i] = self.global_displacements[coord]
            for i, coord in zip(range(3, 6), member.second_node.coord_numbers):
                self.v[member.id, i] = self.global_displacements[coord]
            self.u[member.id] = np.dot(member.T, self.v[member.id])
            member.set_member_end_displacements(self.u[member.id], self.v[member.id])

    def calculate_member_forces(self):
        self.Q = np.zeros([self.model.num_members, 6])
        self.F = np.zeros([self.model.num_members, 6])

        # self.calculate_member_displacements()

        for member in self.model.members:
            self.Q[member.id] = (
                np.dot(member.k, self.u[member.id]) + member.fixed_end_forces_local
            )
            self.F[member.id] = np.dot(member.T.T, self.Q[member.id])
            member.set_member_end_forces(self.Q[member.id], self.F[member.id])

        for member in self.model.members:
            print(f"Membro {member.id}")
            print(member.Q)
            print("\n")

    def calculate_support_reactions(self):
        self.support_reactions = np.zeros([self.model.num_nodes, 4])

        # self.calculate_member_forces()

        for support in self.model.supports:
            for member in self.model.members:
                if member.first_node.id == support.node_id:
                    member.first_node.support_reactions += member.F[:3]
                    self.support_reactions[support.node_id] += np.array(
                        [
                            member.first_node.id,
                            *member.F[:3],
                        ]
                    )
                if member.second_node.id == support.node_id:
                    member.second_node.support_reactions += member.F[3:]
                    self.support_reactions[support.node_id] += np.array(
                        [
                            member.second_node.id,
                            *member.F[:3],
                        ]
                    )

    def solve_frame(self):
        self.calculate_global_displacements()
        self.calculate_member_displacements()
        self.calculate_member_forces()
        self.calculate_support_reactions()


class Results:
    def __init__(self, frame: Frame_2D):
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
        case_index = self.output.index("Caso:")
        num_nodes_index = self.output.index("Número de nós:")
        num_members_index = self.output.index("Número de barras:")

        self.output[case_index] += f" {self.input_file_path.stem}"
        self.output[num_nodes_index] += f" {self.model.num_nodes}"
        self.output[num_members_index] += f" {self.model.num_members}"

    def write_nodes(self):
        row_to_write = self.output.index("Nós:") + 2

        for i, node in enumerate(self.model.nodes):
            self.output.insert(row_to_write, f"{i + 1} | {node.x:6.4E} | {node.y:6.4E}")
            row_to_write += 1

    def write_members(self):
        row_to_write = self.output.index("Barras:") + 2

        for i, member in enumerate(self.model.members):
            self.output.insert(
                row_to_write,
                f"{i + 1} | {member.first_node.id + 1} | {member.second_node.id + 1} | {member.E:6.4E} | {member.A:6.4E} | {member.I:6.4E}",
            )
            row_to_write += 1

    def write_supports(self):
        row_to_write = self.output.index("Apoios:") + 2

        for support in self.model.supports:
            node_id = support.node_id
            dx = "Sim" if support.restrained_coords[0] == 1 else "Não"
            dy = "Sim" if support.restrained_coords[1] == 1 else "Não"
            rz = "Sim" if support.restrained_coords[2] == 1 else "Não"
            self.output.insert(row_to_write, f"{node_id + 1} | {dx} | {dy} | {rz}")
            row_to_write += 1

    def write_applied_node_loads(self):
        row_to_write = self.output.index("Forças nodais:") + 2

        for node in self.model.nodes:
            if node.has_applied_node_load:
                Fx, Fy, Mz = node.applied_node_load.vector

                self.output.insert(row_to_write, f"{node.id + 1} | {Fx} | {Fy} | {Mz}")
                row_to_write += 1

    def write_applied_member_loads(self):
        row_to_write = self.output.index("Cargas distribuídas:") + 2

        for member in self.model.members:
            if member.has_applied_member_load:
                for load in member.applied_member_loads:
                    if "UNIFORM" in load.load_type:
                        load_type = load.load_type.replace("UNIFORM", "UNIFORME")
                    else:
                        load_type = load.load_type
                    self.output.insert(
                        row_to_write,
                        f"{member.id + 1} | {load_type} | {load.w1} | {load.w2}",
                    )
                    row_to_write += 1

    def write_support_reactions(self):
        row_to_write = self.output.index("Reações de apoio:") + 2

        for reaction in self.frame.support_reactions:
            if reaction.any():
                # print(reaction)
                node_id = int(reaction[0])
                FRx, FRy, MRz = reaction[1:]
                self.output.insert(
                    row_to_write,
                    f"{node_id + 1} | {FRx:6.4E} | {FRy:6.4E} | {MRz:6.4E}",
                )
                row_to_write += 1

    def write_member_end_forces(self):
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

    def write_results(self):
        self.write_header()
        self.write_nodes()
        self.write_members()
        self.write_supports()
        self.write_applied_node_loads()
        # self.write_applied_member_loads()
        self.write_support_reactions()
        self.write_member_end_forces()
        print("\n".join(self.output))


if __name__ == "__main__":
    portico_1 = Frame_2D(
        "D:\\repositorios\\structural-matrix-analysis\\Frame\\data\\Exemplo 1 - Pórtico.txt"
    )
    portico_1.solve_frame()
    # results_1 = Results(portico_1)
    # results_1.write_results()
