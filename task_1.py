import copy
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt

# =============================================================================
# 1. Реалізація алгоритму Едмондса-Карпа (максимальний потік)
# =============================================================================

def bfs(capacity, flow, source, sink, parent):
    """
    BFS для знаходження збільшуючого шляху у залишковій мережі.
    
    Args:
        capacity: Матриця пропускної здатності (список списків).
        flow: Матриця потоку (список списків).
        source: Номер вершини-джерела.
        sink: Номер вершини-стоку.
        parent: Список для збереження "батьківських" вузлів на шляху.
    
    Returns:
        True, якщо шлях знайдено, інакше False.
    """
    n = len(capacity)
    visited = [False] * n
    queue = deque([source])
    visited[source] = True
    parent[source] = -1

    while queue:
        u = queue.popleft()
        for v in range(n):
            # Перевіряємо, чи є залишкова пропускна здатність у ребрі u->v
            if not visited[v] and capacity[u][v] - flow[u][v] > 0:
                parent[v] = u
                visited[v] = True
                if v == sink:
                    return True
                queue.append(v)
    return False

def edmonds_karp(capacity, source, sink):
    """
    Реалізація алгоритму Едмондса-Карпа для обчислення максимального потоку.
    
    Args:
        capacity: Матриця пропускної здатності.
        source: Вершина-джерело.
        sink: Вершина-стік.
    
    Returns:
        (max_flow, flow_matrix):
            max_flow - максимальний потік між source і sink,
            flow_matrix - підсумкова матриця потоків.
    """
    n = len(capacity)
    flow_matrix = [[0] * n for _ in range(n)]  # Початково потік нульовий
    parent = [-1] * n
    max_flow = 0

    # Поки існує збільшуючий шлях
    while bfs(capacity, flow_matrix, source, sink, parent):
        # Знаходимо вузьке місце (bottleneck) - мінімальну залишкову пропускну здатність на шляху
        path_flow = float('inf')
        s = sink
        while s != source:
            path_flow = min(path_flow, capacity[parent[s]][s] - flow_matrix[parent[s]][s])
            s = parent[s]
        
        # Оновлюємо потік вздовж знайденого шляху
        v = sink
        while v != source:
            u = parent[v]
            flow_matrix[u][v] += path_flow
            flow_matrix[v][u] -= path_flow  # Зворотний потік
            v = parent[v]
        
        max_flow += path_flow

    return max_flow, flow_matrix

# =============================================================================
# 2. Допоміжні функції для розкладання й аналізу потоку
# =============================================================================

def decompose_flow(flow_matrix, source, sink, terminal_indices, shop_indices):
    """
    Розкладає отриманий потік на зв'язки (Термінал -> Магазин).
    Повертає словник {(terminal, shop): flow_value}.
    
    Args:
        flow_matrix: Підсумкова матриця потоків (результат Едмондса-Карпа).
        source: Індекс суперджерела.
        sink: Індекс суперстоку.
        terminal_indices: Набір індексів, що відповідають терміналам.
        shop_indices: Набір індексів, що відповідають магазинам.
    
    Returns:
        Словник {(terminal, shop): flow_value}, де flow_value - сумарний потік
        від термінала до магазину.
    """
    flow_copy = copy.deepcopy(flow_matrix)
    n = len(flow_copy)

    terminal_shop_flow = {}

    def dfs(u, current_flow, visited, start_terminal=None, end_shop=None):
        if u == sink:
            # Якщо дійшли до стоку, повертаємо знайдений потік,
            # а також який саме термінал та магазин були на шляху.
            return current_flow, start_terminal, end_shop

        visited[u] = True
        for v in range(n):
            if not visited[v] and flow_copy[u][v] > 0:
                # Якщо u є терміналом, фіксуємо його як start_terminal
                new_terminal = start_terminal
                if u in terminal_indices:
                    new_terminal = u
                # Якщо v є магазином, фіксуємо його як end_shop
                new_shop = end_shop
                if v in shop_indices:
                    new_shop = v

                bottleneck = flow_copy[u][v]
                f, t_, s_ = dfs(v, min(current_flow, bottleneck), visited, new_terminal, new_shop)
                if f > 0:
                    flow_copy[u][v] -= f
                    flow_copy[v][u] += f
                    return f, t_, s_
        return 0, None, None

    while True:
        visited = [False] * n
        f, t_, s_ = dfs(source, float('inf'), visited)
        if f == 0:
            break
        if t_ is not None and s_ is not None:
            terminal_shop_flow[(t_, s_)] = terminal_shop_flow.get((t_, s_), 0) + f

    return terminal_shop_flow

def print_flow_table(flow_dict, terminal_map, shop_map):
    """
    Виводить таблицю "Термінал, Магазин, Фактичний Потік" у консоль.

    Args:
        flow_dict: Словник {(terminal, shop): flow_value}.
        terminal_map: Словник {індекс_термінала: "Термінал N"}.
        shop_map: Словник {індекс_магазину: "Магазин M"}.
    """
    print("\nТаблиця потоків між терміналами та магазинами:")
    print("Термінал\tМагазин\t\tФактичний Потік (одиниць)")
    if not flow_dict:
        print("Немає потоку між терміналами та магазинами.")
        return
    for (t, s), flow_val in flow_dict.items():
        t_name = terminal_map.get(t, f"Термінал {t}")
        s_name = shop_map.get(s, f"Магазин {s}")
        print(f"{t_name}\t{s_name}\t\t{flow_val}")

def analyze_results(flow_dict, max_flow):
    """
    Проводить аналіз отриманого потоку:
      1. Які термінали забезпечують найбільший потік товарів до магазинів?
      2. Які маршрути мають найменшу пропускну здатність і як це впливає на загальний потік?
      3. Які магазини отримали найменше товарів і чи можна збільшити їх постачання?
      4. Чи є вузькі місця для покращення ефективності логістичної мережі?
    """
    print("\nАналіз отриманих результатів:")
    print(f"Максимальний потік через мережу: {max_flow} одиниць")

    # Сумарний потік від кожного терміналу та до кожного магазину
    terminal_flow_sum = {}
    shop_flow_sum = {}
    for (t, s), f in flow_dict.items():
        terminal_flow_sum[t] = terminal_flow_sum.get(t, 0) + f
        shop_flow_sum[s] = shop_flow_sum.get(s, 0) + f

    # 1. Найбільший потік із термінала
    if terminal_flow_sum:
        max_t = max(terminal_flow_sum, key=terminal_flow_sum.get)
        print(f"Найбільший потік іде з Терміналу {max_t}, обсяг: {terminal_flow_sum[max_t]} одиниць.")
    else:
        print("Не виявлено потоку з терміналів.")

    # 2. Маршрути з мінімальною пропускною здатністю => потенційні вузькі місця
    print("\nМаршрути з найменшою пропускною здатністю можуть обмежувати загальний потік.")
    print("У наведеній мережі зустрічаються ребра з 5, 10, 15 одиницями тощо - це і є найбільш ймовірні вузькі місця.")

    # 3. Магазин, що отримав найменший обсяг товарів
    if shop_flow_sum:
        min_s = min(shop_flow_sum, key=shop_flow_sum.get)
        print(f"Магазин {min_s} отримав найменше товарів: {shop_flow_sum[min_s]} одиниць.")
        print("Збільшення пропускної здатності шляхів до цього магазину може підвищити його постачання.")
    else:
        print("Жоден магазин не отримав потік ( shop_flow_sum порожній ).")

    # 4. Вузькі місця
    print("\nВузькі місця часто відображені на ребрах із малою пропускною здатністю (5, 10 тощо).")
    print("Якщо збільшити їх, то сумарний потік може зрости, покращуючи ефективність логістики.")

# =============================================================================
# 3. Побудова матриці пропускних здатностей згідно умови
# =============================================================================

def build_capacity_matrix():
    """
    Побудова матриці пропускної здатності для мережі логістики:
    
    Умовно нумеруємо вершини:
      0 - суперджерело
      1 - Термінал 1
      2 - Термінал 2
      3 - Склад 1
      4 - Склад 2
      5 - Склад 3
      6 - Склад 4
      7..20 - Магазини 1..14
      21 - суперстік
    """
    n = 22  # Загальна кількість вершин
    capacity = [[0] * n for _ in range(n)]

    # Суперджерело (0) -> Термінал 1 (1), Термінал 2 (2)
    # Ставимо пропускні здатності = сумі вихідних від терміналів.
    # Термінал 1 -> (Склад1:25 + Склад2:20 + Склад3:15) = 60
    # Термінал 2 -> (Склад3:15 + Склад4:30 + Склад2:10) = 55
    capacity[0][1] = 60
    capacity[0][2] = 55

    # Термінал 1 -> Склади
    capacity[1][3] = 25  # (T1->S1)
    capacity[1][4] = 20  # (T1->S2)
    capacity[1][5] = 15  # (T1->S3)

    # Термінал 2 -> Склади
    capacity[2][5] = 15  # (T2->S3)
    capacity[2][6] = 30  # (T2->S4)
    capacity[2][4] = 10  # (T2->S2)

    # Склад 1 (3) -> Магазини 1..3 (7,8,9)
    capacity[3][7] = 15
    capacity[3][8] = 10
    capacity[3][9] = 20

    # Склад 2 (4) -> Магазини 4..6 (10,11,12)
    capacity[4][10] = 15
    capacity[4][11] = 10
    capacity[4][12] = 25

    # Склад 3 (5) -> Магазини 7..9 (13,14,15)
    capacity[5][13] = 20
    capacity[5][14] = 15
    capacity[5][15] = 10

    # Склад 4 (6) -> Магазини 10..14 (16,17,18,19,20)
    capacity[6][16] = 20
    capacity[6][17] = 10
    capacity[6][18] = 15
    capacity[6][19] = 5
    capacity[6][20] = 10

    # Магазини -> суперстік (21)
    capacity[7][21] = 15   # Магазин 1
    capacity[8][21] = 10   # Магазин 2
    capacity[9][21] = 20   # Магазин 3
    capacity[10][21] = 15  # Магазин 4
    capacity[11][21] = 10  # Магазин 5
    capacity[12][21] = 25  # Магазин 6
    capacity[13][21] = 20  # Магазин 7
    capacity[14][21] = 15  # Магазин 8
    capacity[15][21] = 10  # Магазин 9
    capacity[16][21] = 20  # Магазин 10
    capacity[17][21] = 10  # Магазин 11
    capacity[18][21] = 15  # Магазин 12
    capacity[19][21] = 5   # Магазин 13
    capacity[20][21] = 10  # Магазин 14

    return capacity

# =============================================================================
# 4. Функції для побудови та відображення (networkx)
# =============================================================================

def build_nx_graph(capacity):
    """
    Будує об'єкт графа networkx із заданої матриці пропускної здатності.
    Додає ребра з атрибутом 'capacity'.
    
    ПРИМІТКА: Цей граф використовується лише для візуалізації первинних пропускних здатностей.
              Якщо ви хочете відображати підсумковий потік, потрібно окремо формувати ребра
              з урахуванням flow_matrix.
    """
    G = nx.DiGraph()
    n = len(capacity)

    # Додаємо вершини
    for node in range(n):
        G.add_node(node)

    # Додаємо ребра, де capacity[node_u][node_v] > 0
    for u in range(n):
        for v in range(n):
            if capacity[u][v] > 0:
                G.add_edge(u, v, capacity=capacity[u][v])

    return G

def draw_capacity_graph(G):
    """
    Відображає орієнтований граф G із підписами пропускних здатностей на ребрах.
    Використовує networkx та matplotlib.
    """
    plt.figure(figsize=(10, 6))
    # Для зручного розміщення робимо кілька "шарів"
    # (0 - суперджерело) - зліва,
    # (1..2) - термінали,
    # (3..6) - склади,
    # (7..20) - магазини,
    # (21) - суперстік - справа

    pos = {}
    # суперджерело
    pos[0] = (0, 0)
    # термінали
    pos[1] = (1, 1)
    pos[2] = (1, -1)
    # склади
    pos[3] = (2, 2)
    pos[4] = (2, 0.5)
    pos[5] = (2, -0.5)
    pos[6] = (2, -2)
    # магазини (7..20), розміщуємо їх більш-менш рівномірно
    # візьмемо dx=3, по y робимо від 2 вниз
    shop_y = 2
    for i, node in enumerate(range(7, 21), start=7):
        pos[node] = (3, shop_y)
        shop_y -= 0.3
    # суперстік
    pos[21] = (4, 0)

    # Якщо якась вершина не розміщена, обробимо це (захист від помилок)
    for node in G.nodes():
        if node not in pos:
            pos[node] = (0, 0)  # запасна координата

    # Малюємо граф
    nx.draw_networkx_nodes(G, pos, node_size=1500, node_color="lightblue")
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold")

    # Орієнтовані ребра
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=15)

    # Підписи на ребрах - пропускна здатність
    edge_labels = nx.get_edge_attributes(G, 'capacity')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title("Граф логістичної мережі (Пропускні здатності)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# =============================================================================
# 5. Основна функція для побудови графа, виконання алгоритму та виведення результатів
# =============================================================================

def main():
    # 1. Побудова матриці пропускних здатностей
    capacity_matrix = build_capacity_matrix()

    # 2. Параметри нашого графа
    #    0 - суперджерело, 21 - суперстік
    source = 0
    sink = 21

    # Для зручності задамо словники (карти) індексів у назви терміналів / магазинів
    terminal_map = {
        1: "Термінал 1",
        2: "Термінал 2"
    }
    shop_map = {
        7: "Магазин 1",
        8: "Магазин 2",
        9: "Магазин 3",
        10: "Магазин 4",
        11: "Магазин 5",
        12: "Магазин 6",
        13: "Магазин 7",
        14: "Магазин 8",
        15: "Магазин 9",
        16: "Магазин 10",
        17: "Магазин 11",
        18: "Магазин 12",
        19: "Магазин 13",
        20: "Магазин 14"
    }

    # Списки (або множини) індексів терміналів та магазинів
    terminals = {1, 2}
    shops = {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}

    # 2a. Побудова та відображення графа за допомогою networkx (графіка)
    G = build_nx_graph(capacity_matrix)
    draw_capacity_graph(G)

    # 3. Використовуємо алгоритм Едмондса-Карпа для пошуку максимального потоку
    max_flow, flow_matrix = edmonds_karp(capacity_matrix, source, sink)

    # 4. Виводимо підсумковий максимальний потік
    print(f"Максимальний потік у системі (з {source} до {sink}): {max_flow} одиниць\n")

    # 5. Декомпозиція потоку: скільки з кожного Термінала пішло в кожен Магазин
    flow_dict = decompose_flow(flow_matrix, source, sink, terminals, shops)

    # 6. Виводимо таблицю «Термінал, Магазин, Фактичний Потік»
    print_flow_table(flow_dict, terminal_map, shop_map)

    # 7. Аналіз результатів
    analyze_results(flow_dict, max_flow)

    # 8. Виведемо звіт з розрахунками та поясненнями
    print("\n=== Звіт з розрахунками та поясненнями ===")
    print("1. Побудували модель логістичної мережі, що включає 2 термінали, 4 склади та 14 магазинів,")
    print("   використовуючи матрицю пропускних здатностей (див. build_capacity_matrix).")
    print("2. Використали алгоритм Едмондса-Карпа, який є покращеним варіантом Форда-Фалкерсона.")
    print("3. Отриманий максимальний потік показує, скільки одиниць товару можна доставити від терміналів")
    print("   до магазинів через задані склади з урахуванням обмежень пропускної здатності кожного ребра.")
    print("4. Таблиця «Термінал → Магазин» відображає фактичні обсяги товару, що доходять до магазинів,")
    print("   дає змогу проаналізувати, які термінали і скільки товару постачають.")
    print("5. У висновку про вузькі місця та можливість збільшити пропускну здатність деяких маршрутів,")
    print("   можна підвищити загальний потік і забезпечити кращу логістику товарів.")
    print("6. Результати підтверджують, що за даних обмежень ми досягли оптимального розподілу потоку.")


# =============================================================================
# Запуск основної програми
# =============================================================================
if __name__ == "__main__":
    main()
