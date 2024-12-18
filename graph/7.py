import copy
import heapq
import time
from collections import deque

import networkx as nx
import matplotlib.pyplot as plt

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

    def prim_minimum_spanning_tree(self):


        if not self.adj_list:
            return None

        start_vertex = next(iter(self.adj_list))
        visited = set()
        min_heap = []  #ребра
        mst = []  # мин остов
        total_weight = 0

        for neighbor, weight in self.adj_list[start_vertex]:
            heapq.heappush(min_heap, (weight, start_vertex, neighbor))

        visited.add(start_vertex)

        while min_heap:
            weight, u, v = heapq.heappop(min_heap)  #ребро минимального веса
            if v not in visited:
                visited.add(v)  # + вершина
                mst.append((u, v, weight))  # + ребро
                total_weight += weight

                # + все рёбра из новой вершины
                for neighbor, edge_weight in self.adj_list[v]:
                    if neighbor not in visited:
                        heapq.heappush(min_heap, (edge_weight, v, neighbor))

        return mst, total_weight


    def has_vertex_with_path_sum_leq(self, p):
        if not self.weighted:
            raise ValueError("Граф должен быть взвешенным для выполнения этой операции.")

        def dijkstra(source):
            distances = {v: float('inf') for v in self.adj_list}
            distances[source] = 0
            pq = [(0, source)]
            while pq:
                current_distance, current_vertex = heapq.heappop(pq)
                if current_distance > distances[current_vertex]:
                    continue
                for neighbor, weight in self.adj_list[current_vertex]:
                    distance = current_distance + weight
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        heapq.heappush(pq, (distance, neighbor))
            return distances

        result_distances = {}
        for vertex in self.adj_list:
            distances = dijkstra(vertex)
            result_distances[vertex] = distances
            total_distance = sum(distances[v] for v in distances if distances[v] != float('inf'))
            if total_distance <= p:
                return vertex, result_distances

        return None, result_distances

    def floyd_warshall(self):
        vertices = list(self.adj_list.keys())
        dist = {v: {u: float('inf') for u in vertices} for v in vertices}

        for vertex in vertices:
            dist[vertex][vertex] = 0

        for vertex in self.adj_list:
            for neighbor, weight in self.adj_list[vertex]:
                dist[vertex][neighbor] = weight

        for k in vertices:
            for i in vertices:
                for j in vertices:
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]

        return dist


    def find_negative_weight_cycle(self):
        vertices = list(self.adj_list.keys())
        edges = [(u, v, w) for u in self.adj_list for v, w in self.adj_list[u]]
        dist = {v: float('inf') for v in vertices}
        predecessor = {v: None for v in vertices}

        # Беллман-Форд
        dist[vertices[0]] = 0
        for _ in range(len(vertices) - 1):
            for u, v, w in edges:
                if dist[u] != float('inf') and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    predecessor[v] = u

        # есть ли отрицательный цикл
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                # цикл найден
                cycle = []
                current = v

                # точка входа в цикл
                for _ in range(len(vertices)):
                    current = predecessor[current]

                start_cycle = current
                cycle = [start_cycle]
                current = predecessor[start_cycle]

                while current != start_cycle:
                    cycle.append(current)
                    current = predecessor[current]

                cycle.append(start_cycle)
                cycle.reverse()

                return cycle

        return None

    def find_max_flow(self, source, sink):
        # алгоритм Форда-Фалкерсона с поиском пути увеличения с помощью обхода в ширину

        vertices = list(self.adj_list.keys())
        residual_capacity = {u: {v: 0 for v in vertices} for u in vertices}

        # остаточные пропускные способности
        for u in self.adj_list:
            for v, capacity in self.adj_list[u]:
                residual_capacity[u][v] = capacity

        parent = {v: None for v in vertices}
        max_flow = 0

        def bfs(source, sink):
            visited = {v: False for v in vertices}
            queue = [source]
            visited[source] = True

            while queue:
                current = queue.pop(0)
                for neighbor in residual_capacity[current]:
                    if not visited[neighbor] and residual_capacity[current][neighbor] > 0:
                        queue.append(neighbor)
                        visited[neighbor] = True
                        parent[neighbor] = current
                        if neighbor == sink:
                            return True
            return False

        # путь увеличения
        while bfs(source, sink):
            # минимальная остаточная пропускная способность на пути
            path_flow = float('inf')
            current = sink
            while current != source:
                prev = parent[current]
                path_flow = min(path_flow, residual_capacity[prev][current])
                current = prev

            # обновление остаточных пропускных способностей
            current = sink
            while current != source:
                prev = parent[current]
                residual_capacity[prev][current] -= path_flow
                residual_capacity[current][prev] += path_flow
                current = prev

            # увеличение общего потока
            max_flow += path_flow

        return max_flow

    def visualize_graph(self, highlight=None, pos=None, pause_interval=1):
        """
        Визуализация графа с пошаговым обновлением.
        :param highlight: Словарь с посещёнными вершинами и рёбрами.
        :param pos: Позиции узлов для визуализации.
        :param pause_interval: Задержка между шагами.
        """
        G = nx.DiGraph() if self.directed else nx.Graph()

        # Добавляем рёбра
        for vertex, neighbors in self.adj_list.items():
            for neighbor in neighbors:
                if isinstance(neighbor, tuple):  # Взвешенный граф
                    G.add_edge(vertex, neighbor[0], weight=neighbor[1])
                else:
                    G.add_edge(vertex, neighbor)

        plt.clf()  # Очищаем текущее окно
        edge_labels = nx.get_edge_attributes(G, "weight")

        # Рисуем граф
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=10)
        if self.weighted:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        # Подсветка узлов и рёбер
        if highlight:
            nx.draw_networkx_nodes(G, pos, nodelist=highlight["nodes"], node_color='yellow', node_size=2500)
            nx.draw_networkx_edges(G, pos, edgelist=highlight["edges"], edge_color='blue', width=2)

        plt.pause(pause_interval)  # Пауза для обновления визуализации

    def bfs_highlight(self, start_vertex, delay=1):
        """
        Алгоритм обхода в ширину с возвратом информации для визуализации.
        :param start_vertex: Начальная вершина.
        :param delay: Задержка между шагами в секундах.
        :return: Словарь с посещёнными вершинами и рёбрами.
        """
        if start_vertex not in self.adj_list:
            raise ValueError(f"Вершина '{start_vertex}' отсутствует в графе.")

        visited = set()
        queue = deque([start_vertex])
        highlight = {"nodes": [], "edges": []}
        pos = nx.spring_layout(self.adj_list)  # Генерация позиций узлов один раз перед началом

        try:
            plt.ion()  # Включаем интерактивный режим Matplotlib
            while queue:
                current = queue.popleft()
                if current not in visited:
                    visited.add(current)
                    highlight["nodes"].append(current)

                    for neighbor in self.adj_list[current]:
                        if isinstance(neighbor, tuple):  # Если граф взвешенный
                            neighbor = neighbor[0]
                        if neighbor not in visited:
                            queue.append(neighbor)
                            highlight["edges"].append((current, neighbor))

                    # Пошаговая визуализация графа
                    self.visualize_graph(highlight=highlight, pos=pos, pause_interval=delay)

        except KeyboardInterrupt:
            print("Процесс визуализации был прерван пользователем.")
        finally:
            plt.ioff()
            plt.show()

    def dfs_highlight(self, start_vertex, delay=1):
        """
        Алгоритм обхода в глубину с возвратом информации для визуализации.
        :param start_vertex: Начальная вершина.
        :param delay: Задержка между шагами в секундах.
        :return: Словарь с посещёнными вершинами и рёбрами.
        """
        if start_vertex not in self.adj_list:
            raise ValueError(f"Вершина '{start_vertex}' отсутствует в графе.")

        visited = set()
        highlight = {"nodes": [], "edges": []}
        pos = nx.spring_layout(self.adj_list)  # Генерация позиций узлов один раз перед началом

        def dfs(v):
            # Отмечаем вершину
            visited.add(v)
            highlight["nodes"].append(v)

            # Визуализируем после посещения вершины
            self.visualize_graph(highlight=highlight, pos=pos, pause_interval=delay)

            # Обходим соседей
            for neighbor in self.adj_list[v]:
                if isinstance(neighbor, tuple):  # Если граф взвешенный
                    neighbor = neighbor[0]
                if neighbor not in visited:
                    highlight["edges"].append((v, neighbor))  # Добавляем ребро
                    dfs(neighbor)  # Рекурсивный вызов для соседа

        try:
            plt.ion()  # Включаем интерактивный режим Matplotlib
            self.visualize_graph(highlight=None)  # Сбрасываем все выделения перед началом обхода
            dfs(start_vertex)  # Начинаем обход с начальной вершины

        except KeyboardInterrupt:
            print("Процесс визуализации был прерван пользователем.")
        finally:
            plt.ioff()  # Отключаем интерактивный режим
            plt.show()  # Окно останется открытым, пока пользователь не закроет его вручную


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
    print("13. Алгоритм Прима")
    print("14. Минимальная сумма путей < P")
    print("15. Кратчайшие пути для всех пар вершин.")
    print("16. Цикл отрицательного веса, если он есть")
    print("17. Максимальный поток")
    print("18. Визуализировать граф")
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


def handle_find_minimum_spanning_tree(graphs):
    graph = choose_graph(graphs)
    if graph:
        if not graph.directed:
            try:
                mst_edges, total_weight = graph.prim_minimum_spanning_tree()  # Поиск остова
                if mst_edges:
                    print("Минимальный остов графа (рёбра и их веса):")
                    for edge in mst_edges:
                        print(f"{edge[0]} - {edge[1]} (вес: {edge[2]})")
                    print(f"Общий вес остова: {total_weight}")
                else:
                    print("Не удалось найти минимальный остов. Возможно, граф несвязен.")
            except ValueError as e:
                print(f"Ошибка при вычислении минимального остова: {e}")
        else:
            print("Ошибка: Алгоритм Прима работает только для неориентированных графов.")


def handle_check_vertex_path_sum(graphs):
    graph = choose_graph(graphs)
    if graph:
        try:
            p = int(input("Введите значение P: "))
            vertex, distances = graph.has_vertex_with_path_sum_leq(p)
            if vertex:
                total_distance = sum(
                    distances[vertex][v] for v in distances[vertex] if distances[vertex][v] != float('inf'))
                print(f"Вершина {vertex} имеет суммарную стоимость путей до всех остальных не более {p}.")
                print(f"Сумма расстояний для вершины {vertex}: {total_distance}")
            else:
                print(f"Нет вершины, суммарная стоимость путей от которой до остальных не превосходит {p}.")


        except ValueError as e:
            print(f"Ошибка: {e}")


def handle_show_shortest_paths(graphs):
    graph = choose_graph(graphs)
    if not graph:
        return

    dist = graph.floyd_warshall()

    print("Кратчайшие пути для всех пар вершин:")
    for vertex in dist:
        for target in dist[vertex]:
            if dist[vertex][target] == float('inf'):
                print(f"Путь из {vertex} в {target}: нет пути")
            else:
                print(f"Путь из {vertex} в {target}: {dist[vertex][target]}")


def handle_find_negative_cycle(graphs):
    graph = choose_graph(graphs)
    if graph is None:
        print("Граф не выбран.")
        return

    try:
        cycle = graph.find_negative_weight_cycle()
        if cycle:
            print("Найден отрицательный цикл:")
            print(" -> ".join(map(str, cycle)))
        else:
            print("Отрицательных циклов в графе не найдено.")
    except Exception as e:
        print(f"Ошибка при поиске отрицательного цикла: {e}")

def handle_max_flow(graphs):
    graph = choose_graph(graphs)
    if graph is None:
        print("Граф не выбран.")
        return

    try:
        source = input("Введите вершину-источник: ").strip()
        sink = input("Введите вершину-сток: ").strip()

        if source not in graph.adj_list or sink not in graph.adj_list:
            print("Ошибка: указанные вершины отсутствуют в графе.")
            return

        max_flow = graph.find_max_flow(source, sink)
        print(f"Максимальный поток из {source} в {sink}: {max_flow}")
    except Exception as e:
        print(f"Ошибка при нахождении максимального потока: {e}")

def handle_visualize_graph(graphs):
    graph = choose_graph(graphs)
    if graph is None:
        print("Граф не выбран.")
        return

    print("Выберите тип визуализации:")
    print("1. Обычная визуализация")
    print("2. Визуализация пути")
    print("3. Визуализация алгоритма обхода в ширину")
    print("4. Визуализация алгоритма обхода в глубину")  # Новый пункт

    choice = input("Введите номер выбора: ").strip()

    if choice == "1":
        try:
            graph.visualize_graph()
        except Exception as e:
            print(f"Ошибка при визуализации графа: {e}")
    elif choice == "2":
        path_input = input("Введите путь через пробел (например, S A B T): ").strip()
        if not path_input:
            print("Ошибка: путь не указан.")
            return
        path = path_input.split()
        try:
            graph.visualize_graph(path=path)
        except Exception as e:
            print(f"Ошибка при визуализации пути: {e}")
    elif choice == "3":
        start_vertex = input("Введите стартовую вершину для обхода в ширину: ").strip()
        if not start_vertex:
            print("Ошибка: стартовая вершина не указана.")
            return
        try:
            delay_input = input("Укажите задержку между шагами в секундах (по умолчанию 1): ").strip()
            delay = float(delay_input) if delay_input else 1.0
            highlight = graph.bfs_highlight(start_vertex, delay=delay)  # Передаём задержку в BFS
            graph.visualize_graph(highlight=highlight)
        except ValueError:
            print("Ошибка: задержка должна быть числом.")
        except Exception as e:
            print(f"Ошибка при выполнении алгоритма: {e}")
    elif choice == "4":  # Обработка выбора DFS
        start_vertex = input("Введите стартовую вершину для обхода в глубину: ").strip()
        if not start_vertex:
            print("Ошибка: стартовая вершина не указана.")
            return
        try:
            delay_input = input("Укажите задержку между шагами в секундах (по умолчанию 1): ").strip()
            delay = float(delay_input) if delay_input else 1.0
            highlight = graph.dfs_highlight(start_vertex, delay=delay)  # Передаём задержку в DFS
            graph.visualize_graph(highlight=highlight)
        except ValueError:
            print("Ошибка: задержка должна быть числом.")
        except Exception as e:
            print(f"Ошибка при выполнении алгоритма: {e}")
    else:
        print("Неверный выбор.")










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
        elif choice == "13":
            handle_find_minimum_spanning_tree(graphs)
        elif choice == "14":
            handle_check_vertex_path_sum(graphs)
        elif choice == "15":
            handle_show_shortest_paths(graphs)
        elif choice == "16":
            handle_find_negative_cycle(graphs)
        elif choice == "17":
            handle_max_flow(graphs)
        elif choice == "18":
            handle_visualize_graph(graphs)
        elif choice == "0":
            print("Выход из программы")
            break
        else:
            print("Такого пункта нет")


if __name__ == "__main__":
    main()

