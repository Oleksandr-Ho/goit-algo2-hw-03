# Завдання 1: Застосування алгоритму максимального потоку для логістики товарів

У цій задачі потрібно було побудувати модель логістичної мережі (2 термінали, 4 склади, 14 магазинів) та розрахувати максимальний потік від терміналів до магазинів з урахуванням пропускної здатності ребер. Реалізація виконана за допомогою алгоритму Едмондса-Карпа.

## 1. Огляд

- **Модель**:  
  - Штучна вершина **0** — суперджерело (s), з якого «витікає» потік у термінали.  
  - Термінали **1** та **2** — початкові вузли (джерела товарів).  
  - Склади **3**, **4**, **5**, **6** — проміжні вузли.  
  - Магазини **7**...**20** — кінцеві вузли, які отримують товар.  
  - Штучна вершина **21** — суперстік (t), куди надходить увесь потік від магазинів.  

- **Алгоритм**: Використано **Едмондса-Карпа** (удосконалений варіант Форда-Фалкерсона), який гарантує поліноміальний час виконання завдяки пошуку в ширину (BFS) для пошуку збільшуючих шляхів.

## 2. Візуалізація графа

Нижче наведено орієнтовне зображення мережі, де:  
- Вершина **0** — суперджерело (штучна),  
- Вершини **1** і **2** — термінали,  
- Вершини **3**, **4**, **5**, **6** — склади,  
- Вершини **7**...**20** — магазини (14 штук),  
- Вершина **21** — суперстік (штучна).

![alt text](<Граф логістичної меражі.PNG>)

На графі зображені орієнтовані ребра з відповідною пропускною здатністю (capacity). Зокрема, видно:
- **0 → 1** (суперджерело до Термінала 1),  
- **0 → 2** (суперджерело до Термінала 2),  
- Далі ребра від терміналів до складів,  
- Від складів до магазинів,  
- І, насамкінець, від магазинів до суперстоку (21).

## 3. Результати виконання (лог консольного виведення)

Максимальний потік у системі (з 0 до 21): 115 одиниць


Таблиця потоків між терміналами та магазинами:
Термінал        Магазин         Фактичний Потік (одиниць)
Термінал 1      Магазин 1               15
Термінал 1      Магазин 2               10
Термінал 1      Магазин 4               15
Термінал 1      Магазин 5               5
Термінал 1      Магазин 7               15
Термінал 2      Магазин 5               5
Термінал 2      Магазин 6               5
Термінал 2      Магазин 7               5
Термінал 2      Магазин 8               10
Термінал 2      Магазин 10              20
Термінал 2      Магазин 11              10

## 4. Аналіз отриманих результатів
Аналіз отриманих результатів: Максимальний потік через мережу: 115 одиниць найбільший потік іде з Терміналу 1, обсяг: 60 одиниць.
Маршрути з найменшою пропускною здатністю можуть обмежувати загальний потік. 
У наведеній мережі зустрічаються ребра з 5, 10, 15 одиницями тощо - це і є найбільш ймовірні вузькі місця. 
Магазин 12 отримав найменше товарів: 5 одиниць. Збільшення пропускної здатності шляхів до цього магазину може підвищити його постачання.
Вузькі місця часто відображені на ребрах із малою пропускною здатністю (5, 10 тощо). 
Якщо збільшити їх, то сумарний потік може зрости, покращуючи ефективність логістики.


## 5. Пояснення та відповіді на запитання

### 5.1. Які термінали забезпечують найбільший потік товарів до магазинів?

Згідно з підсумком:
- **Термінал 1** (в нашому випадку індекс 1) передає найбільший потік — **60 одиниць**.  
- **Термінал 2** дає 55 одиниць.  

Таким чином, Термінал 1 — лідер із постачання товарів.

### 5.2. Які маршрути мають найменшу пропускну здатність і як це впливає на загальний потік?

Найнижчі пропускні здатності (5, 10 тощо) зустрічаються, наприклад:
- **(Склад 4 → Магазин 13)** пропускна здатність 5,  
- **(Термінал 2 → Склад 2)** має 10,  
- **(Термінал 1 → Склад 3)** 15 тощо.  

Вони є потенційними «вузькими місцями»: як тільки на шляху трапляється ребро з маленькою capacity, воно обмежує потік усього шляху, адже пропускна здатність шляху визначається мінімальним ребром.

### 5.3. Які магазини отримали найменше товарів і чи можна збільшити їх постачання?

У результатах видно, що:
- **Магазин 12** отримав лише 5 одиниць (найнижчий показник).  

Якщо збільшити пропускну здатність ребер, що ведуть до цього магазину (наприклад, від Складу 2 або 4, залежно від мережі), теоретично можна досягти більшого потоку до нього — звісно, якщо це не впливає негативно на інші вузькі місця в мережі.

### 5.4. Чи є вузькі місця, які можна усунути для покращення ефективності логістичної мережі?

Так, ребра з пропускною здатністю 5 чи 10 одиниць:
- Це «вузьке місце», яке потенційно зменшує загальну пропускну спроможність мережі.  
- Збільшення capacity таких ребер може підвищити сумарний потік і дозволити більше товару доставляти до магазинів.



# Завдання 2: Порівняння ефективності OOBTree та dict для діапазонних запитів

Цей проект демонструє порівняння продуктивності діапазонних запитів на даних про товари, збережених в OOBTree (з бібліотеки BTrees) та стандартному словнику Python (dict).

## Опис проекту

Проект завантажує набір даних з CSV файлу де надано 100 000 записів.

Кожен рядок файлу містить інформацію про товар з наступними полями:
- **ID**: Унікальний ідентифікатор товару.
- **Name**: Назва товару.
- **Category**: Категорія товару.
- **Price**: Ціна товару.

Дані зберігаються в двох структурах:
1. **OOBTree** – використовується ключ у вигляді кортежу **(Price, ID)**, що забезпечує автоматичне сортування за ціною та дозволяє ефективно виконувати діапазонні запити.
2. **Dict** – стандартний словник, де ключем є **ID**, а діапазонний запит виконується через лінійний пошук.

## Перевага використання ключа (Price, ID) в OOBTree

Для структури OOBTree було обрано ключ у вигляді кортежу **(Price, ID)** замість використання лише **ID**. Це забезпечує:
- **Відсортованість за ціною:** Товари автоматично впорядковуються за зростанням ціни.
- **Оптимізацію діапазонних запитів:** Метод `items(lower_bound, upper_bound)` дозволяє швидко отримати всі товари в заданому діапазоні цін, що значно покращує продуктивність запитів.

## Результати продуктивності

При виконанні 100 діапазонних запитів отримано наступні результати:

- **OOBTree:** 0.136129 seconds  
- **Dict:** 1.592360 seconds

Як видно з результатів, використання OOBTree з ключем **(Price, ID)** дозволило суттєво знизити час виконання діапазонних запитів у порівнянні зі стандартним словником.