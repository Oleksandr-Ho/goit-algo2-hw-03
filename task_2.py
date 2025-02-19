import csv
import timeit
from BTrees.OOBTree import OOBTree

def add_item_to_tree(tree, item):
    """
    Додає товар до OOBTree.
    Ключ формується як (Price, ID) для забезпечення відсортованості за ціною.
    """
    # Перетворення даних: ID в int, Price в float
    item_id = int(item["ID"])
    price = float(item["Price"])
    # Формуємо ключ як кортеж (price, id)
    key = (price, item_id)
    tree[key] = item

def add_item_to_dict(d, item):
    """
    Додає товар до словника dict.
    Ключем є ID, значенням – словник з атрибутами товару.
    """
    item_id = int(item["ID"])
    # Переконуємось, що ціна збережена як float
    item["Price"] = float(item["Price"])
    d[item_id] = item

def range_query_tree(tree, low, high):
    """
    Виконує діапазонний запит по OOBTree для знаходження всіх товарів,
    у яких ціна знаходиться в межах [low, high].
    Використовується метод items(lower_bound, upper_bound).
    """
    lower_bound = (low, -float('inf'))
    upper_bound = (high, float('inf'))
    # Повертаємо список товарів, що потрапляють у зазначений діапазон
    return [value for key, value in tree.items(lower_bound, upper_bound)]

def range_query_dict(d, low, high):
    """
    Виконує діапазонний запит по словнику через лінійний пошук.
    Повертає товари, у яких значення Price знаходиться між low та high.
    """
    return [item for item in d.values() if low <= item["Price"] <= high]

def load_data(filepath):
    """
    Завантажує дані з CSV файлу та додає кожен рядок (товар)
    до OOBTree та словника dict.
    """
    tree = OOBTree()
    data_dict = {}
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            add_item_to_tree(tree, row)
            add_item_to_dict(data_dict, row)
    return tree, data_dict

def main():
    # Вказаний шлях до файлу CSV
    filepath = r"goit-algo2-hw-03\generated_items_data.csv"
    print("Завантаження даних...")
    tree, data_dict = load_data(filepath)
    print("Дані завантажено.")

    # Задаємо діапазон запиту по ціні
    query_low = 100.0
    query_high = 200.0

    # Виконуємо 100 діапазонних запитів та вимірюємо час виконання для OOBTree
    tree_time = timeit.timeit(
        lambda: range_query_tree(tree, query_low, query_high), number=100
    )

    # Виконуємо 100 діапазонних запитів та вимірюємо час виконання для словника dict
    dict_time = timeit.timeit(
        lambda: range_query_dict(data_dict, query_low, query_high), number=100
    )

    print(f"Total range_query time for OOBTree: {tree_time:.6f} seconds")
    print(f"Total range_query time for Dict: {dict_time:.6f} seconds")

if __name__ == '__main__':
    main()
