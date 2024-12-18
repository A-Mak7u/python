import copy


class Graph:
    def __init__(self, directed=False, weighted=False, filename=None):
        self.adj_list = {}
        self.directed = directed
        self.weighted = weighted

        if filename:
            self.load_from_file(filename)

    def add_vertex(self, vertex):
        if vertex not in self.adj_list:
            self.adj_list[vertex] = []

    def add_edge(self, vertex1, vertex2, weight=None):
        if vertex1 not in self.adj_list or vertex2 not in self.adj_list:
            print(f"Ошибка: одна или обе вершины {vertex1} или {vertex2} не существуют.")
            return

        if self.weighted:
            if any(neighbor == (vertex2, weight) for neighbor in self.adj_list[vertex1]):
                print(f"Ребро {vertex1} -> {vertex2} с весом {weight} уже существует.")
                return
            elif any(neighbor[0] == vertex2 for neighbor in self.adj_list[vertex1]):
                print(f"Ошибка: Ребро {vertex1} -> {vertex2} уже существует с другим весом.")
                return
        else:
            if vertex2 in self.adj_list[vertex1] or (not self.directed and vertex1 in self.adj_list[vertex2]):
                print(f"Ребро {vertex1} -> {vertex2} уже существует.")
                return

        if self.weighted:
            self.adj_list[vertex1].append((vertex2, weight))
        else:
            self.adj_list[vertex1].append(vertex2)

        if not self.directed and vertex1 != vertex2:
            if self.weighted:
                self.adj_list[vertex2].append((vertex1, weight))
            else:
                self.adj_list[vertex2].append(vertex1)

        print(f"Ребро добавлено между {vertex1} и {vertex2}.")

    def remove_vertex(self, vertex):
        if vertex in self.adj_list:
            del self.adj_list[vertex]

        for neighbors in self.adj_list.values():
            for n in neighbors[:]:
                if n == vertex or (isinstance(n, tuple) and n[0] == vertex):
                    neighbors.remove(n)

    def remove_edge(self, vertex1, vertex2):
        edge_exists = False

        if vertex1 in self.adj_list:
            for neighbor in self.adj_list[vertex1][:]:
                if (isinstance(neighbor, tuple) and neighbor[0] == vertex2) or neighbor == vertex2:
                    self.adj_list[vertex1].remove(neighbor)
                    edge_exists = True
                    break

        if not self.directed and vertex2 in self.adj_list:
            for neighbor in self.adj_list[vertex2][:]:
                if (isinstance(neighbor, tuple) and neighbor[0] == vertex1) or neighbor == vertex1:
                    self.adj_list[vertex2].remove(neighbor)
                    edge_exists = True
                    break

        if edge_exists:
            print(f"Ребро между {vertex1} и {vertex2} удалено")
        else:
            print(f"Ошибка: Ребро между {vertex1} и {vertex2} не существует")

        if vertex1 in self.adj_list and not self.adj_list[vertex1]:
            self.adj_list[vertex1] = []
        if vertex2 in self.adj_list and not self.adj_list[vertex2]:
            self.adj_list[vertex2] = []

    def display_info(self):
        print(f"Тип графа: {'Ориентированный' if self.directed else 'Неориентированный'}")
        print(f"Взвешенный: {'Да' if self.weighted else 'Нет'}")
        print("Список смежности:")
        for vertex, neighbors in self.adj_list.items():
            if self.weighted:
                neighbors_str = ", ".join(f"{n[0]} (вес: {n[1]})" for n in neighbors)
            else:
                neighbors_str = ", ".join(str(n) for n in neighbors)
            print(f"{vertex}: {neighbors_str if neighbors_str else ''}")

    #5
    def is_tree_or_forest(self):
        def is_connected_and_acyclic(start, visited, parent=None):
            visited.add(start)
            for neighbor in self.adj_list.get(start, []):
                if isinstance(neighbor, tuple):
                    neighbor = neighbor[0]
                if neighbor not in visited:
                    if not is_connected_and_acyclic(neighbor, visited, start):
                        return False
                elif neighbor != parent:
                    return False
            return True

        visited = set()
        components = 0

        for vertex in self.adj_list:
            if vertex not in visited:
                components += 1
                if not is_connected_and_acyclic(vertex, visited):
                    return "Граф не является ни деревом, ни лесом."

        if components == 1:
            return "Граф является деревом."
        return "Граф является лесом."

    #6
    def find_common_vertex_with_equal_paths(self, u, v):
        if u not in self.adj_list or v not in self.adj_list:
            return f"Ошибка: одна или обе вершины ({u}, {v}) отсутствуют в графе."

        def bfs_distances_by_edges(start):
            distances = {start: 0}
            queue = [start]
            while queue:
                current = queue.pop(0)
                for neighbor in self.adj_list[current]:
                    if isinstance(neighbor, tuple):  # Если граф взвешенный, берём только вершину
                        neighbor = neighbor[0]
                    if neighbor not in distances:
                        distances[neighbor] = distances[current] + 1
                        queue.append(neighbor)
            return distances

        distances_u = bfs_distances_by_edges(u)
        distances_v = bfs_distances_by_edges(v)

        common_vertices = [
            vertex for vertex in distances_u
            if vertex in distances_v and distances_u[vertex] == distances_v[vertex]
        ]

        if not common_vertices:
            return "Нет таких вершин."

        return f"Вершины с равными длинами путей из {u} и {v}: {', '.join(common_vertices)}"

    def save_to_file(self, filename):
        if not filename.endswith('.txt'):
            filename += '.txt'

        with open(filename, 'w') as f:
            f.write(f"{'Directed' if self.directed else 'Undirected'}\n")
            f.write(f"{'Yes' if self.weighted else 'No'}\n")
            written_edges = set()
            for vertex, neighbors in self.adj_list.items():
                if not neighbors:
                    f.write(f"{vertex}\n")
                for neighbor in neighbors:
                    if self.weighted:
                        edge = (vertex, neighbor[0]) if vertex <= neighbor[0] else (neighbor[0], vertex)
                        if edge not in written_edges:
                            f.write(f"{vertex} {neighbor[0]} {neighbor[1]}\n")
                            written_edges.add(edge)
                    else:
                        edge = (vertex, neighbor) if vertex <= neighbor else (neighbor, vertex)
                        if edge not in written_edges:
                            f.write(f"{vertex} {neighbor}\n")
                            written_edges.add(edge)

    def load_from_file(self, filename):
        if not filename.endswith('.txt'):
            filename += '.txt'

        with open(filename, 'r') as f:
            lines = f.readlines()
            self.directed = "Directed" in lines[0]
            self.weighted = "Yes" in lines[1]

            for line in lines[2:]:
                data = line.strip().split()
                if len(data) == 1:
                    vertex = data[0]
                    self.add_vertex(vertex)
                elif len(data) == 2 or len(data) == 3:
                    vertex1 = data[0]
                    vertex2 = data[1]
                    self.add_vertex(vertex1)
                    self.add_vertex(vertex2)
                    if self.weighted and len(data) == 3:
                        weight = int(data[2])
                        self.add_edge(vertex1, vertex2, weight)
                    else:
                        self.add_edge(vertex1, vertex2)

    def copy(self):
        new_graph = Graph(directed=self.directed, weighted=self.weighted)
        new_graph.adj_list = copy.deepcopy(self.adj_list)
        return new_graph

    def remove_isolated_vertices(self):
        all_vertices = set(self.adj_list.keys())
        non_isolated = set()

        for vertex, neighbors in self.adj_list.items():
            if neighbors:
                non_isolated.add(vertex)
            non_isolated.update(neighbors)

        final_isolated = list(all_vertices - non_isolated)

        for vertex in final_isolated:
            del self.adj_list[vertex]

        return final_isolated


def print_menu():
    print("\nМеню:")
    print("1. Новый граф")
    print("2. Добавить вершину в граф")
    print("3. Добавить ребро в граф")
    print("4. Удалить вершину из графа")
    print("5. Удалить ребро из графа")
    print("6. Показать информацию о графе")
    print("7. Сохранить граф в файл")
    print("8. Загрузить граф из файла")
    print("9. Создать копию графа")
    print("10. Проверить, является ли граф деревом или лесом")
    print("11. Найти вершину, в которую есть пути одинаковой длины из двух вершин.")
    print("12. Удалить изолированные вершины")
    print("0. Выйти")


def choose_graph(graphs):
    if not graphs:
        print("Нет доступных графов.")
        return None

    print("Доступные графы:")
    for i, graph_name in enumerate(graphs.keys(), start=1):
        print(f"{i}. {graph_name}")

    choice = input("Выберите граф по номеру: ").strip()
    try:
        index = int(choice) - 1
        graph_name = list(graphs.keys())[index]
        return graphs[graph_name]
    except (IndexError, ValueError):
        print("Ошибка: Некорректный выбор.")
        return None


def handle_add_graph(graphs):
    graph_name = input("Введите имя для нового графа: ").strip()
    if graph_name in graphs:
        print(f"Граф с именем {graph_name} уже существует.")
        return

    directed_choice = input("Ориентированный граф (y/n)? ").strip().lower() == 'y'
    weighted_choice = input("Взвешенный граф (y/n)? ").strip().lower() == 'y'

    graphs[graph_name] = Graph(directed=directed_choice, weighted=weighted_choice)
    print(f"Граф {graph_name} создан")


def handle_add_vertex(graphs):
    graph = choose_graph(graphs)
    if graph:
        vertex = input("Введите вершину (буква): ").strip()
        if not vertex.isalpha() or len(vertex) > 1:
            print("Ошибка: Введите одну букву.")
            return
        if vertex in graph.adj_list:
            print(f"Вершина {vertex} уже существует.")
        else:
            graph.add_vertex(vertex)
            print(f"Вершина {vertex} добавлена.")


def handle_add_edge(graphs):
    graph = choose_graph(graphs)
    if graph:
        vertex1 = input("Введите первую вершину: ").strip()
        vertex2 = input("Введите вторую вершину: ").strip()

        if vertex1 not in graph.adj_list or vertex2 not in graph.adj_list:
            print(f"Ошибка: Одна или обе вершины не существуют.")
            return

        if graph.weighted:
            try:
                weight = int(input("Введите вес ребра: ").strip())
                graph.add_edge(vertex1, vertex2, weight)
            except ValueError:
                print("Ошибка: Введите целое число для веса.")
        else:
            graph.add_edge(vertex1, vertex2)


def handle_remove_vertex(graphs):
    graph = choose_graph(graphs)
    if graph:
        vertex = input("Введите вершину для удаления: ").strip()
        if vertex in graph.adj_list:
            graph.remove_vertex(vertex)
            print(f"Вершина {vertex} удалена.")
        else:
            print(f"Ошибка: Вершина {vertex} не найдена.")


def handle_remove_edge(graphs):
    graph = choose_graph(graphs)
    if graph:
        vertex1 = input("Введите первую вершину: ").strip()
        vertex2 = input("Введите вторую вершину: ").strip()

        if vertex1 in graph.adj_list and vertex2 in graph.adj_list:
            graph.remove_edge(vertex1, vertex2)
        else:
            print(f"Ошибка: Одна или обе вершины не существуют.")


def handle_display_info(graphs):
    graph = choose_graph(graphs)
    if graph:
        graph.display_info()


def handle_save_graph(graphs):
    graph = choose_graph(graphs)
    if graph:
        filename = input("Введите имя файла для сохранения: ").strip()
        graph.save_to_file(filename)
        print(f"Граф сохранён в файл {filename}.")


def handle_load_graph(graphs):
    filename = input("Введите имя файла для загрузки: ").strip()
    graph_name = input("Введите имя для загруженного графа: ").strip()

    if graph_name in graphs:
        print(f"Граф с именем {graph_name} уже существует.")
    else:
        try:
            graphs[graph_name] = Graph(filename=filename)
            print(f"Граф {graph_name} загружен из файла {filename}")
        except FileNotFoundError:
            print(f"Ошибка: Файл {filename} не найден")


def handle_copy_graph(graphs):
    graph = choose_graph(graphs)
    if graph:
        copy_name = input("Введите имя для копии графа: ").strip()
        if copy_name in graphs:
            print(f"Граф с именем {copy_name} уже существует")
        else:
            graphs[copy_name] = graph.copy()
            print(f"Копия графа создана с именем {copy_name}.")


def handle_check_tree_or_forest(graphs):
    graph = choose_graph(graphs)
    if graph:
        result = graph.is_tree_or_forest()
        print(result)

def handle_find_common_vertex(graphs):
    graph = choose_graph(graphs)
    if graph:
        u = input("Введите первую вершину: ").strip()
        v = input("Введите вторую вершину: ").strip()

        if u in graph.adj_list and v in graph.adj_list:
            result = graph.find_common_vertex_with_equal_paths(u, v)
            print(result)
        else:
            print("Ошибка: Одна или обе вершины не существуют в графе.")

def handle_remove_isolated_vertices(graphs):
    graph = choose_graph(graphs)
    if graph:
        removed_vertices = graph.remove_isolated_vertices()
        if removed_vertices:
            print(f"Изолированные вершины удалены: {', '.join(removed_vertices)}")
        else:
            print("Изолированных вершин не найдено.")


def main():
    graphs = {}

    while True:
        print_menu()
        choice = input("Введите номер действия: ").strip()

        if choice == "1":
            handle_add_graph(graphs)
        elif choice == "2":
            handle_add_vertex(graphs)
        elif choice == "3":
            handle_add_edge(graphs)
        elif choice == "4":
            handle_remove_vertex(graphs)
        elif choice == "5":
            handle_remove_edge(graphs)
        elif choice == "6":
            handle_display_info(graphs)
        elif choice == "7":
            handle_save_graph(graphs)
        elif choice == "8":
            handle_load_graph(graphs)
        elif choice == "9":
            handle_copy_graph(graphs)
        elif choice == "10":
            handle_check_tree_or_forest(graphs)
        elif choice == "11":
            handle_find_common_vertex(graphs)
        elif choice == "12":
            handle_remove_isolated_vertices(graphs)
        elif choice == "0":
            print("Выход из программы")
            break
        else:
            print("Такого пункта нет")


if __name__ == "__main__":
    main()

