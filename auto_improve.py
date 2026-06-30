"""
═══════════════════════════════════════════════════════════════════
  Stars AI — نظام التحسين التلقائي
  يحلّل نتائج الـ Benchmark → يكتشف الفئات الضعيفة
  → يولّد بيانات تدريب مخصصة → يعيد التدريب → يقيّم من جديد

  التشغيل:
    python auto_improve.py --model ./models/stars_expert_merged
    python auto_improve.py --gguf  ./models/stars_expert.gguf
    python auto_improve.py --model ./models/stars_expert_merged --rounds 3
    python auto_improve.py --model ./models/stars_expert_merged --from-report benchmark.json
═══════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import QUESTIONS, ModelEvaluator, GGUFEngine, HFEngine, print_report


# ════════════════════════════════════════════════════════════════════
# بيانات التحسين المخصصة لكل فئة
# ════════════════════════════════════════════════════════════════════

BOOST_DATA = {

    "Python أساسي": [
        {"instruction": "اكتب دالة تُعيد عكس قائمة", "output": "def reverse_list(lst):\n    return lst[::-1]\n\n# أو:\ndef reverse_list(lst):\n    return list(reversed(lst))\n\nprint(reverse_list([1,2,3,4,5]))  # [5,4,3,2,1]"},
        {"instruction": "اكتب دالة تُزيل المكررات من قائمة مع الحفاظ على الترتيب", "output": "def remove_duplicates(lst):\n    seen = set()\n    result = []\n    for item in lst:\n        if item not in seen:\n            seen.add(item)\n            result.append(item)\n    return result\n\nprint(remove_duplicates([1,2,2,3,1,4]))  # [1,2,3,4]"},
        {"instruction": "اكتب دالة تحسب المتوسط الحسابي لقائمة أرقام", "output": "def average(numbers):\n    if not numbers:\n        return 0\n    return sum(numbers) / len(numbers)\n\nprint(average([10, 20, 30, 40]))  # 25.0"},
        {"instruction": "اكتب دالة تُبدّل قيمتين بدون متغير مؤقت", "output": "def swap(a, b):\n    a, b = b, a\n    return a, b\n\nx, y = 5, 10\nx, y = swap(x, y)\nprint(x, y)  # 10 5"},
        {"instruction": "اكتب دالة تُحوّل درجات فهرنهايت إلى سيلسيوس", "output": "def fahrenheit_to_celsius(f):\n    return (f - 32) * 5 / 9\n\nprint(fahrenheit_to_celsius(100))  # 37.77..."},
        {"instruction": "اكتب دالة تتحقق إذا كانت قائمة مرتبة تصاعدياً", "output": "def is_sorted(lst):\n    return all(lst[i] <= lst[i+1] for i in range(len(lst)-1))\n\nprint(is_sorted([1,2,3,4]))   # True\nprint(is_sorted([1,3,2,4]))   # False"},
        {"instruction": "اكتب دالة تُنسّق رقم هاتف بصيغة (XXX) XXX-XXXX", "output": "def format_phone(digits):\n    d = ''.join(filter(str.isdigit, digits))\n    return f'({d[:3]}) {d[3:6]}-{d[6:10]}'\n\nprint(format_phone('1234567890'))  # (123) 456-7890"},
        {"instruction": "اكتب دالة تُعيد أكثر عنصر تكراراً في قائمة", "output": "def most_frequent(lst):\n    return max(set(lst), key=lst.count)\n\nprint(most_frequent([1,2,2,3,2,4]))  # 2"},
        {"instruction": "اكتب برنامج يطبع مثلث نجوم من 1 إلى N", "output": "def star_triangle(n):\n    for i in range(1, n+1):\n        print('*' * i)\n\nstar_triangle(5)\n# *\n# **\n# ***\n# ****\n# *****"},
        {"instruction": "اكتب دالة تُعيد جميع أزواج الأرقام التي مجموعها يساوي target", "output": "def find_pairs(lst, target):\n    pairs = []\n    seen = set()\n    for n in lst:\n        complement = target - n\n        if complement in seen:\n            pairs.append((complement, n))\n        seen.add(n)\n    return pairs\n\nprint(find_pairs([1,2,3,4,5,6], 7))  # [(3,4),(2,5),(1,6)]"},
        {"instruction": "اكتب دالة تُحوّل نص إلى camelCase", "output": "def to_camel_case(text):\n    words = text.replace('-', ' ').replace('_', ' ').split()\n    return words[0].lower() + ''.join(w.capitalize() for w in words[1:])\n\nprint(to_camel_case('hello world'))        # helloWorld\nprint(to_camel_case('some-mixed_string'))  # someMixedString"},
        {"instruction": "اكتب دالة تُولّد كلمة مرور عشوائية", "output": "import random\nimport string\n\ndef generate_password(length=12):\n    chars = string.ascii_letters + string.digits + '!@#$%'\n    return ''.join(random.choice(chars) for _ in range(length))\n\nprint(generate_password(16))"},
    ],

    "OOP": [
        {"instruction": "اكتب class BankAccount مع deposit وwithdraw وbalance", "output": "class BankAccount:\n    def __init__(self, owner, balance=0):\n        self.owner = owner\n        self._balance = balance\n\n    def deposit(self, amount):\n        if amount <= 0:\n            raise ValueError('المبلغ يجب أن يكون موجباً')\n        self._balance += amount\n        return self._balance\n\n    def withdraw(self, amount):\n        if amount > self._balance:\n            raise ValueError('رصيد غير كافٍ')\n        self._balance -= amount\n        return self._balance\n\n    @property\n    def balance(self):\n        return self._balance\n\n    def __str__(self):\n        return f'حساب {self.owner}: {self._balance}'\n\nacc = BankAccount('علي', 1000)\nacc.deposit(500)\nacc.withdraw(200)\nprint(acc)  # حساب علي: 1300"},
        {"instruction": "اكتب abstract class Shape مع دوال area وperimeter", "output": "from abc import ABC, abstractmethod\nimport math\n\nclass Shape(ABC):\n    @abstractmethod\n    def area(self): pass\n\n    @abstractmethod\n    def perimeter(self): pass\n\n    def describe(self):\n        return f'{type(self).__name__}: area={self.area():.2f}, perimeter={self.perimeter():.2f}'\n\nclass Circle(Shape):\n    def __init__(self, r): self.r = r\n    def area(self): return math.pi * self.r ** 2\n    def perimeter(self): return 2 * math.pi * self.r\n\nclass Rectangle(Shape):\n    def __init__(self, w, h): self.w, self.h = w, h\n    def area(self): return self.w * self.h\n    def perimeter(self): return 2 * (self.w + self.h)\n\nfor s in [Circle(5), Rectangle(3, 4)]:\n    print(s.describe())"},
        {"instruction": "اكتب Singleton pattern في Python", "output": "class Singleton:\n    _instance = None\n\n    def __new__(cls, *args, **kwargs):\n        if not cls._instance:\n            cls._instance = super().__new__(cls)\n        return cls._instance\n\n    def __init__(self, value=None):\n        if value is not None:\n            self.value = value\n\ns1 = Singleton(42)\ns2 = Singleton()\nprint(s1 is s2)    # True\nprint(s2.value)    # 42"},
        {"instruction": "اكتب Observer pattern في Python", "output": "class EventEmitter:\n    def __init__(self):\n        self._listeners = {}\n\n    def on(self, event, callback):\n        self._listeners.setdefault(event, []).append(callback)\n\n    def emit(self, event, *args):\n        for cb in self._listeners.get(event, []):\n            cb(*args)\n\n# مثال\nemitter = EventEmitter()\nemitter.on('data', lambda x: print(f'مستمع 1: {x}'))\nemitter.on('data', lambda x: print(f'مستمع 2: {x*2}'))\nemitter.emit('data', 10)\n# مستمع 1: 10\n# مستمع 2: 20"},
        {"instruction": "اكتب dataclass للمنتج مع التحقق من الصحة", "output": "from dataclasses import dataclass, field\nfrom typing import Optional\n\n@dataclass\nclass Product:\n    name: str\n    price: float\n    quantity: int = 0\n    category: Optional[str] = None\n\n    def __post_init__(self):\n        if self.price < 0:\n            raise ValueError('السعر لا يمكن أن يكون سالباً')\n        if self.quantity < 0:\n            raise ValueError('الكمية لا يمكن أن تكون سالبة')\n\n    @property\n    def total_value(self):\n        return self.price * self.quantity\n\np = Product('لابتوب', 2500.0, 10, 'إلكترونيات')\nprint(p.total_value)  # 25000.0"},
        {"instruction": "اكتب mixin class لإضافة serialization لأي class", "output": "import json\n\nclass SerializableMixin:\n    def to_dict(self):\n        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}\n\n    def to_json(self):\n        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)\n\n    @classmethod\n    def from_dict(cls, data):\n        obj = cls.__new__(cls)\n        obj.__dict__.update(data)\n        return obj\n\nclass User(SerializableMixin):\n    def __init__(self, name, email):\n        self.name = name\n        self.email = email\n\nu = User('محمد', 'mo@example.com')\nprint(u.to_json())"},
    ],

    "خوارزميات": [
        {"instruction": "اكتب Quick Sort في Python", "output": "def quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left  = [x for x in arr if x < pivot]\n    mid   = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quick_sort(left) + mid + quick_sort(right)\n\nprint(quick_sort([3,6,8,10,1,2,1]))  # [1,1,2,3,6,8,10]"},
        {"instruction": "اكتب خوارزمية Dijkstra لأقصر مسار", "output": "import heapq\n\ndef dijkstra(graph, start):\n    dist = {node: float('inf') for node in graph}\n    dist[start] = 0\n    pq = [(0, start)]\n\n    while pq:\n        d, u = heapq.heappop(pq)\n        if d > dist[u]:\n            continue\n        for v, w in graph[u]:\n            if dist[u] + w < dist[v]:\n                dist[v] = dist[u] + w\n                heapq.heappush(pq, (dist[v], v))\n    return dist\n\ngraph = {\n    'A': [('B',1),('C',4)],\n    'B': [('C',2),('D',5)],\n    'C': [('D',1)],\n    'D': []\n}\nprint(dijkstra(graph, 'A'))  # {'A':0,'B':1,'C':3,'D':4}"},
        {"instruction": "اكتب دالة تحل مسألة Knapsack (حقيبة الظهر) بالبرمجة الديناميكية", "output": "def knapsack(weights, values, capacity):\n    n = len(weights)\n    dp = [[0]*(capacity+1) for _ in range(n+1)]\n\n    for i in range(1, n+1):\n        for w in range(capacity+1):\n            dp[i][w] = dp[i-1][w]\n            if weights[i-1] <= w:\n                dp[i][w] = max(dp[i][w],\n                               dp[i-1][w-weights[i-1]] + values[i-1])\n    return dp[n][capacity]\n\nweights = [2, 3, 4, 5]\nvalues  = [3, 4, 5, 6]\nprint(knapsack(weights, values, 8))  # 10"},
        {"instruction": "اكتب خوارزمية DFS على شجرة ثنائية", "output": "class TreeNode:\n    def __init__(self, val=0, left=None, right=None):\n        self.val, self.left, self.right = val, left, right\n\ndef inorder(root):\n    return inorder(root.left) + [root.val] + inorder(root.right) if root else []\n\ndef preorder(root):\n    return [root.val] + preorder(root.left) + preorder(root.right) if root else []\n\ndef postorder(root):\n    return postorder(root.left) + postorder(root.right) + [root.val] if root else []\n\nroot = TreeNode(1, TreeNode(2, TreeNode(4), TreeNode(5)), TreeNode(3))\nprint(inorder(root))   # [4,2,5,1,3]\nprint(preorder(root))  # [1,2,4,5,3]"},
        {"instruction": "اكتب دالة تجد أطول سلسلة جزئية مشتركة (LCS)", "output": "def lcs(s1, s2):\n    m, n = len(s1), len(s2)\n    dp = [[0]*(n+1) for _ in range(m+1)]\n    for i in range(1, m+1):\n        for j in range(1, n+1):\n            if s1[i-1] == s2[j-1]:\n                dp[i][j] = dp[i-1][j-1] + 1\n            else:\n                dp[i][j] = max(dp[i-1][j], dp[i][j-1])\n    return dp[m][n]\n\nprint(lcs('ABCBDAB', 'BDCABA'))  # 4"},
        {"instruction": "اكتب دالة تتحقق من دورة في Linked List (Floyd's Algorithm)", "output": "class Node:\n    def __init__(self, val):\n        self.val  = val\n        self.next = None\n\ndef has_cycle(head):\n    slow = fast = head\n    while fast and fast.next:\n        slow = slow.next\n        fast = fast.next.next\n        if slow is fast:\n            return True\n    return False\n\n# إنشاء قائمة مرتبطة بدورة\nn1, n2, n3 = Node(1), Node(2), Node(3)\nn1.next = n2; n2.next = n3; n3.next = n2  # دورة\nprint(has_cycle(n1))  # True"},
    ],

    "Python متقدم": [
        {"instruction": "اكتب context manager باستخدام contextlib", "output": "from contextlib import contextmanager\nimport time\n\n@contextmanager\ndef timer(label=''):\n    start = time.perf_counter()\n    try:\n        yield\n    finally:\n        elapsed = time.perf_counter() - start\n        print(f'{label}: {elapsed:.4f}s')\n\nwith timer('حساب'):\n    total = sum(range(1_000_000))\nprint(total)"},
        {"instruction": "اكتب LRU Cache من الصفر بدون functools", "output": "from collections import OrderedDict\n\nclass LRUCache:\n    def __init__(self, capacity):\n        self.cap   = capacity\n        self.cache = OrderedDict()\n\n    def get(self, key):\n        if key not in self.cache:\n            return -1\n        self.cache.move_to_end(key)\n        return self.cache[key]\n\n    def put(self, key, value):\n        if key in self.cache:\n            self.cache.move_to_end(key)\n        self.cache[key] = value\n        if len(self.cache) > self.cap:\n            self.cache.popitem(last=False)\n\ncache = LRUCache(3)\ncache.put(1, 'a'); cache.put(2, 'b'); cache.put(3, 'c')\nprint(cache.get(1))  # a\ncache.put(4, 'd')    # يحذف المفتاح 2\nprint(cache.get(2))  # -1"},
        {"instruction": "اكتب async generator للبيانات المتدفقة", "output": "import asyncio\n\nasync def stream_numbers(n, delay=0.1):\n    for i in range(n):\n        await asyncio.sleep(delay)\n        yield i * i\n\nasync def main():\n    async for value in stream_numbers(5):\n        print(f'قيمة: {value}')\n\nasyncio.run(main())\n# قيمة: 0\n# قيمة: 1\n# قيمة: 4\n# قيمة: 9\n# قيمة: 16"},
        {"instruction": "اكتب pipeline باستخدام generators", "output": "def read_data(items):\n    yield from items\n\ndef filter_even(stream):\n    for x in stream:\n        if x % 2 == 0:\n            yield x\n\ndef square(stream):\n    for x in stream:\n        yield x ** 2\n\ndef take(stream, n):\n    for i, x in enumerate(stream):\n        if i >= n: break\n        yield x\n\n# Pipeline\ndata     = read_data(range(100))\nevens    = filter_even(data)\nsquared  = square(evens)\nresult   = list(take(squared, 5))\nprint(result)  # [0, 4, 16, 36, 64]"},
        {"instruction": "اكتب decorator يُعيد المحاولة تلقائياً عند الفشل (retry)", "output": "import time\nimport functools\n\ndef retry(max_attempts=3, delay=1.0, exceptions=(Exception,)):\n    def decorator(func):\n        @functools.wraps(func)\n        def wrapper(*args, **kwargs):\n            for attempt in range(1, max_attempts+1):\n                try:\n                    return func(*args, **kwargs)\n                except exceptions as e:\n                    if attempt == max_attempts:\n                        raise\n                    print(f'محاولة {attempt} فشلت: {e}. إعادة بعد {delay}s...')\n                    time.sleep(delay)\n        return wrapper\n    return decorator\n\n@retry(max_attempts=3, delay=0.5)\ndef unstable_function():\n    import random\n    if random.random() < 0.7:\n        raise ConnectionError('فشل الاتصال')\n    return 'نجح!'"},
        {"instruction": "اكتب Protocol (interface) في Python باستخدام typing", "output": "from typing import Protocol, runtime_checkable\n\n@runtime_checkable\nclass Drawable(Protocol):\n    def draw(self) -> str: ...\n    def resize(self, factor: float) -> None: ...\n\nclass Circle:\n    def __init__(self, r): self.r = r\n    def draw(self): return f'دائرة r={self.r}'\n    def resize(self, f): self.r *= f\n\nclass Square:\n    def __init__(self, s): self.s = s\n    def draw(self): return f'مربع s={self.s}'\n    def resize(self, f): self.s *= f\n\ndef render(shape: Drawable):\n    print(shape.draw())\n\nfor s in [Circle(5), Square(3)]:\n    render(s)\n    print(isinstance(s, Drawable))  # True"},
    ],

    "قواعد البيانات": [
        {"instruction": "اكتب ORM بسيط لإدارة قاعدة بيانات SQLite", "output": "import sqlite3\nfrom typing import List, Optional\n\nclass Model:\n    table = ''\n    db    = 'app.db'\n\n    @classmethod\n    def _conn(cls):\n        return sqlite3.connect(cls.db)\n\n    @classmethod\n    def all(cls) -> List[dict]:\n        with cls._conn() as conn:\n            conn.row_factory = sqlite3.Row\n            rows = conn.execute(f'SELECT * FROM {cls.table}').fetchall()\n            return [dict(r) for r in rows]\n\n    @classmethod\n    def find(cls, id: int) -> Optional[dict]:\n        with cls._conn() as conn:\n            conn.row_factory = sqlite3.Row\n            row = conn.execute(f'SELECT * FROM {cls.table} WHERE id=?', (id,)).fetchone()\n            return dict(row) if row else None\n\n    @classmethod\n    def create(cls, **kwargs):\n        cols = ', '.join(kwargs.keys())\n        vals = ', '.join('?' * len(kwargs))\n        with cls._conn() as conn:\n            cur = conn.execute(f'INSERT INTO {cls.table} ({cols}) VALUES ({vals})',\n                               list(kwargs.values()))\n            conn.commit()\n            return cur.lastrowid\n\nclass User(Model):\n    table = 'users'"},
        {"instruction": "اكتب class لقراءة وكتابة ملفات JSON مع تسجيل التاريخ", "output": "import json\nimport os\nfrom datetime import datetime\n\nclass JsonStorage:\n    def __init__(self, filepath):\n        self.filepath = filepath\n        self._data = self._load()\n\n    def _load(self):\n        if os.path.exists(self.filepath):\n            with open(self.filepath, encoding='utf-8') as f:\n                return json.load(f)\n        return {}\n\n    def _save(self):\n        with open(self.filepath, 'w', encoding='utf-8') as f:\n            json.dump(self._data, f, ensure_ascii=False, indent=2)\n\n    def set(self, key, value):\n        self._data[key] = {'value': value, 'updated_at': datetime.now().isoformat()}\n        self._save()\n\n    def get(self, key, default=None):\n        entry = self._data.get(key)\n        return entry['value'] if entry else default\n\n    def delete(self, key):\n        self._data.pop(key, None)\n        self._save()\n\nstorage = JsonStorage('store.json')\nstorage.set('user', {'name': 'علي'})\nprint(storage.get('user'))"},
        {"instruction": "اكتب دالة تُصدّر قاعدة بيانات SQLite إلى CSV", "output": "import sqlite3\nimport csv\n\ndef export_table_to_csv(db_path, table, csv_path):\n    with sqlite3.connect(db_path) as conn:\n        cursor = conn.execute(f'SELECT * FROM {table}')\n        columns = [d[0] for d in cursor.description]\n        rows    = cursor.fetchall()\n\n    with open(csv_path, 'w', newline='', encoding='utf-8') as f:\n        writer = csv.writer(f)\n        writer.writerow(columns)\n        writer.writerows(rows)\n\n    print(f'✓ صُدِّر {len(rows)} صف إلى {csv_path}')\n\nexport_table_to_csv('app.db', 'users', 'users.csv')"},
        {"instruction": "اكتب migration system بسيط لـ SQLite", "output": "import sqlite3\n\nMIGRATIONS = [\n    ('001_create_users',\n     'CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT, email TEXT UNIQUE)'),\n    ('002_add_created_at',\n     'ALTER TABLE users ADD COLUMN created_at TEXT DEFAULT CURRENT_TIMESTAMP'),\n    ('003_create_posts',\n     'CREATE TABLE IF NOT EXISTS posts (id INTEGER PRIMARY KEY, user_id INTEGER, content TEXT)'),\n]\n\ndef run_migrations(db_path):\n    with sqlite3.connect(db_path) as conn:\n        conn.execute('CREATE TABLE IF NOT EXISTS migrations (name TEXT PRIMARY KEY)')\n        done = {r[0] for r in conn.execute('SELECT name FROM migrations').fetchall()}\n\n        for name, sql in MIGRATIONS:\n            if name not in done:\n                conn.execute(sql)\n                conn.execute('INSERT INTO migrations VALUES (?)', (name,))\n                print(f'  ✓ {name}')\n            else:\n                print(f'  — {name} (منفّذة مسبقاً)')\n        conn.commit()\n\nrun_migrations('app.db')"},
        {"instruction": "اكتب rate limiter لطلبات API", "output": "import time\nfrom collections import deque\n\nclass RateLimiter:\n    def __init__(self, max_calls, period):\n        self.max_calls = max_calls\n        self.period    = period\n        self.calls     = deque()\n\n    def __call__(self, func):\n        import functools\n        @functools.wraps(func)\n        def wrapper(*args, **kwargs):\n            now = time.time()\n            # حذف الطلبات القديمة\n            while self.calls and now - self.calls[0] > self.period:\n                self.calls.popleft()\n            if len(self.calls) >= self.max_calls:\n                sleep_for = self.period - (now - self.calls[0])\n                print(f'  انتظار {sleep_for:.2f}s...')\n                time.sleep(sleep_for)\n            self.calls.append(time.time())\n            return func(*args, **kwargs)\n        return wrapper\n\n@RateLimiter(max_calls=3, period=1.0)\ndef api_call(n):\n    print(f'طلب #{n}')\n\nfor i in range(6):\n    api_call(i)"},
    ],
}


# ════════════════════════════════════════════════════════════════════
# التحليل — اكتشاف الفئات الضعيفة
# ════════════════════════════════════════════════════════════════════

WEAK_THRESHOLD  = 55.0   # أقل من 55% = فئة ضعيفة
TARGET_SCORE    = 70.0   # الهدف بعد التحسين

@dataclass
class WeakCategory:
    name:       str
    pass_rate:  float
    gap:        float          # الفجوة حتى الهدف
    priority:   int = 1        # كلما كان أقل كلما كانت الأولوية أعلى


def analyze_weaknesses(evaluation_summary: dict) -> list[WeakCategory]:
    """يحلّل نتائج التقييم ويُحدد الفئات التي تحتاج تحسيناً."""
    weak = []
    for cat, data in evaluation_summary.get("by_category", {}).items():
        rate = data.get("pass_rate", 0)
        if rate < TARGET_SCORE:
            gap = TARGET_SCORE - rate
            weak.append(WeakCategory(cat, rate, gap))

    # ترتيب حسب الضعف (الأضعف أولاً)
    weak.sort(key=lambda w: w.pass_rate)
    for i, w in enumerate(weak):
        w.priority = i + 1
    return weak


def print_analysis(weak: list[WeakCategory], before_rate: float):
    """يطبع تقرير التحليل."""
    print(f"\n{'═'*56}")
    print(f"  تحليل نقاط الضعف")
    print(f"{'═'*56}")
    print(f"  نسبة النجاح الحالية : {before_rate:.1f}%")
    print(f"  الهدف المستهدف      : {TARGET_SCORE:.1f}%")

    if not weak:
        print(f"\n  ✓ نموذجك ممتاز! كل الفئات تتجاوز {TARGET_SCORE}%")
        return

    print(f"\n  الفئات التي تحتاج تحسيناً:")
    print(f"  {'─'*50}")
    for w in weak:
        bar_fill = int(w.pass_rate / 5)
        bar_gap  = int(w.gap / 5)
        bar = "█" * bar_fill + "░" * bar_gap + " " * (20 - bar_fill - bar_gap)
        print(f"  #{w.priority}  {w.name:20s} [{bar}] {w.pass_rate:.1f}% (نقص {w.gap:.1f}%)")
    print(f"  {'─'*50}")


# ════════════════════════════════════════════════════════════════════
# توليد بيانات التحسين
# ════════════════════════════════════════════════════════════════════

def generate_boost_data(weak_categories: list[WeakCategory], output_path: str, multiplier: int = 3) -> int:
    """
    يولّد بيانات تدريب مخصصة للفئات الضعيفة.
    الفئات الأضعف تأخذ بيانات أكثر (multiplier × priority).
    """
    print(f"\n{'═'*56}")
    print(f"  توليد بيانات التحسين المخصصة")
    print(f"{'═'*56}")

    all_samples = []
    for w in weak_categories:
        cat_data = BOOST_DATA.get(w.name, [])
        if not cat_data:
            print(f"  ⚠ لا توجد بيانات جاهزة لـ: {w.name}")
            continue

        # كلما كانت الفئة أضعف، كلما أعطيناها بيانات أكثر
        repeats = max(1, multiplier + (len(weak_categories) - w.priority))
        samples = cat_data * repeats
        all_samples.extend(samples)
        print(f"  ✓ {w.name:22s} ← {len(samples)} مثال (×{repeats})")

    import random
    random.shuffle(all_samples)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n  ✓ الإجمالي: {len(all_samples)} مثال → {output_path}")
    return len(all_samples)


# ════════════════════════════════════════════════════════════════════
# إعادة التدريب على الفئات الضعيفة
# ════════════════════════════════════════════════════════════════════

def finetune_on_weak(base_model_path: str, data_path: str, output_dir: str) -> str:
    """يُعيد التدريب على البيانات المحسّنة."""
    try:
        import torch
        from datasets import Dataset
        from transformers import (
            AutoTokenizer, AutoModelForCausalLM,
            TrainingArguments, Trainer,
            DataCollatorForLanguageModeling,
        )
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError as e:
        print(f"❌ مكتبة مفقودة: {e}")
        sys.exit(1)

    print(f"\n{'═'*56}")
    print(f"  إعادة التدريب على الفئات الضعيفة")
    print(f"{'═'*56}")
    print(f"  النموذج   : {base_model_path}")
    print(f"  البيانات  : {data_path}")
    print(f"  المخرج    : {output_dir}")

    PROMPT = "### المهمة:\n{instruction}\n\n### الكود:\n{output}"
    MAX_LEN = 512

    # تحميل البيانات
    samples = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except Exception:
                    pass
    print(f"  ← {len(samples)} مثال")

    # تحميل النموذج
    print(f"\n  تحميل النموذج...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, device_map="auto", trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    # LoRA مع معدل تعلم أعلى للتحسين السريع
    lora_cfg = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)

    # Tokenization
    texts = [PROMPT.format(**s) for s in samples if "instruction" in s and "output" in s]

    def tokenize(batch):
        result = tokenizer(batch["text"], max_length=MAX_LEN,
                           truncation=True, padding="max_length")
        result["labels"] = result["input_ids"].copy()
        return result

    dataset = Dataset.from_dict({"text": texts})
    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

    os.makedirs(output_dir, exist_ok=True)

    # التدريب — epoch أقل لأنه تحسين وليس تدريباً من الصفر
    args = TrainingArguments(
        output_dir=output_dir, num_train_epochs=2,
        per_device_train_batch_size=4, gradient_accumulation_steps=4,
        learning_rate=3e-4, warmup_ratio=0.1, lr_scheduler_type="cosine",
        logging_steps=10, save_steps=100, save_total_limit=1,
        fp16=torch.cuda.is_available(), optim="adamw_torch", report_to="none",
    )

    trainer = Trainer(
        model=model, args=args, train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()

    # الحفظ
    print(f"\n  حفظ النموذج المحسّن...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    merged_dir = output_dir + "_merged"
    merged = model.merge_and_unload()
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    print(f"  ✓ محفوظ: {merged_dir}")
    return merged_dir


# ════════════════════════════════════════════════════════════════════
# التقييم المقارن
# ════════════════════════════════════════════════════════════════════

def evaluate_model(engine, label: str, questions: list) -> dict:
    """يُقيّم نموذجاً ويعيد ملخص النتائج."""
    evaluator = ModelEvaluator(engine, label)
    return evaluator.run(questions, verbose=False)


def print_improvement_report(
    before: dict,
    after:  dict,
    round_num: int,
    weak_cats: list[WeakCategory],
):
    """يطبع تقرير التحسين قبل وبعد."""
    diff_overall = after["pass_rate"] - before["pass_rate"]

    print(f"\n{'╔'+'═'*58+'╗'}")
    print(f"║  تقرير التحسين — الجولة #{round_num}{'':44}║")
    print(f"{'╠'+'═'*58+'╣'}")
    print(f"║  {'المقياس':20s} {'قبل':>10s} {'بعد':>10s} {'الفرق':>10s}  ║")
    print(f"{'╠'+'─'*58+'╣'}")
    print(f"║  {'نسبة النجاح الكلية':20s} {before['pass_rate']:>9.1f}% {after['pass_rate']:>9.1f}% {diff_overall:>+9.1f}%  ║")
    print(f"║  {'متوسط الدقة':20s} {before['avg_score']:>9.1f}% {after['avg_score']:>9.1f}% {after['avg_score']-before['avg_score']:>+9.1f}%  ║")
    print(f"{'╠'+'─'*58+'╣'}")
    print(f"║  الفئات المستهدفة:{'':41}║")

    for w in weak_cats:
        b = before["by_category"].get(w.name, {}).get("pass_rate", 0)
        a = after["by_category"].get(w.name, {}).get("pass_rate", 0)
        d = a - b
        arrow = "↑" if d > 0 else ("↓" if d < 0 else "→")
        color = "✓" if a >= TARGET_SCORE else "•"
        print(f"║  {color} {w.name:20s} {b:>7.1f}%  →  {a:>7.1f}%  {arrow}{abs(d):>5.1f}%   ║")

    print(f"{'╠'+'═'*58+'╣'}")

    if diff_overall >= 15:
        verdict = "تحسّن رائع! الجولة نجحت بامتياز"
    elif diff_overall >= 8:
        verdict = "تحسّن جيد — استمر لجولة أخرى"
    elif diff_overall >= 3:
        verdict = "تحسّن بسيط — أضف بيانات أكثر"
    else:
        verdict = "تحسّن ضئيل — جرّب زيادة epochs"
    print(f"║  {verdict:<58}║")
    print(f"{'╚'+'═'*58+'╝'}\n")


# ════════════════════════════════════════════════════════════════════
# نقطة الدخول
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Stars AI — نظام التحسين التلقائي",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
أمثلة:
  # تشغيل دورة تحسين واحدة
  python auto_improve.py --model ./models/stars_expert_merged

  # 3 جولات تحسين متتالية
  python auto_improve.py --model ./models/stars_expert_merged --rounds 3

  # من ملف GGUF
  python auto_improve.py --gguf ./models/stars_expert.gguf

  # من تقرير benchmark موجود مسبقاً
  python auto_improve.py --model ./models/stars_expert_merged --from-report benchmark.json

  # تحسين فئة محددة فقط
  python auto_improve.py --model ./models/stars_expert_merged --category "خوارزميات"
        """,
    )
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--model",        help="مسار النموذج (HuggingFace)")
    g.add_argument("--gguf",         help="مسار ملف GGUF")
    parser.add_argument("--rounds",      type=int, default=1, help="عدد جولات التحسين (افتراضي: 1)")
    parser.add_argument("--from-report", help="ملف JSON من benchmark مسبق (لتخطّي التقييم الأولي)")
    parser.add_argument("--category",    help="تحسين فئة واحدة فقط")
    parser.add_argument("--output",      default=None, help="مجلد المخرجات")
    args = parser.parse_args()

    is_gguf     = bool(args.gguf)
    model_path  = args.gguf or args.model
    model_name  = os.path.basename(model_path.rstrip("/\\"))
    base_output = args.output or os.path.join(
        os.path.dirname(model_path.rstrip("/\\")), "improved"
    )
    os.makedirs(base_output, exist_ok=True)

    # تصفية الأسئلة
    questions = QUESTIONS
    if args.category:
        questions = [q for q in QUESTIONS if args.category in q["category"]]

    print(f"\n{'═'*60}")
    print(f"  Stars AI — نظام التحسين التلقائي")
    print(f"{'═'*60}")
    print(f"  النموذج  : {model_name}")
    print(f"  الجولات  : {args.rounds}")
    print(f"  المخرج   : {base_output}")

    # ── التقييم الأولي ────────────────────────────────────────────
    if args.from_report:
        print(f"\n  [التقييم الأولي] من ملف: {args.from_report}")
        with open(args.from_report, encoding="utf-8") as f:
            before_summary = json.load(f)
            # دعم كلا صيغتي: مباشر أو داخل leaderboard
            if "leaderboard" in before_summary:
                my_entry = next(
                    (e for e in before_summary["leaderboard"] if e.get("is_mine")),
                    before_summary["leaderboard"][0],
                )
                before_summary = {
                    "pass_rate":   my_entry["pass_rate"],
                    "avg_score":   my_entry["avg_score"],
                    "avg_time_s":  my_entry.get("avg_time_s", 0),
                    "by_category": my_entry.get("by_category", {}),
                }
    else:
        print(f"\n  [التقييم الأولي] تقييم النموذج الحالي...")
        engine = GGUFEngine(model_path) if is_gguf else HFEngine(model_path)
        before_summary = evaluate_model(engine, model_name, questions)
        print_report(before_summary)

    # حفظ التقييم الأولي
    initial_path = os.path.join(base_output, "eval_before.json")
    with open(initial_path, "w", encoding="utf-8") as f:
        json.dump(before_summary, f, ensure_ascii=False, indent=2)

    # ── جولات التحسين ─────────────────────────────────────────────
    current_model  = model_path
    current_is_gguf = is_gguf
    prev_summary    = before_summary
    history         = [before_summary]

    for round_num in range(1, args.rounds + 1):
        print(f"\n{'░'*60}")
        print(f"  الجولة {round_num} / {args.rounds}")
        print(f"{'░'*60}")

        # 1. تحليل الضعف
        weak = analyze_weaknesses(prev_summary)
        if args.category:
            weak = [w for w in weak if args.category in w.name]

        print_analysis(weak, prev_summary["pass_rate"])

        if not weak:
            print(f"  🎉 النموذج حقق الهدف المطلوب في كل الفئات!")
            break

        # 2. توليد بيانات التحسين
        boost_path = os.path.join(base_output, f"boost_round{round_num}.jsonl")
        count = generate_boost_data(weak, boost_path, multiplier=3)

        if count == 0:
            print("  ⚠ لا توجد بيانات للتحسين — توقف")
            break

        if current_is_gguf:
            print("\n  ⚠ لا يمكن إعادة تدريب ملفات GGUF مباشرة.")
            print("  → تأكد من وجود النموذج بصيغة HuggingFace أولاً:")
            print(f"    python auto_improve.py --model ./models/stars_expert_merged --rounds {args.rounds}")
            break

        # 3. إعادة التدريب
        round_output = os.path.join(base_output, f"round{round_num}")
        improved_model = finetune_on_weak(current_model, boost_path, round_output)

        # 4. التقييم بعد التحسين
        print(f"\n  [تقييم] بعد الجولة #{round_num}...")
        engine_after = HFEngine(improved_model)
        after_summary = evaluate_model(engine_after, f"Round{round_num}", questions)

        # 5. تقرير التحسين
        print_improvement_report(prev_summary, after_summary, round_num, weak)

        # حفظ
        after_path = os.path.join(base_output, f"eval_round{round_num}.json")
        with open(after_path, "w", encoding="utf-8") as f:
            json.dump(after_summary, f, ensure_ascii=False, indent=2)

        history.append(after_summary)
        prev_summary  = after_summary
        current_model = improved_model

    # ── الملخص النهائي ────────────────────────────────────────────
    final = history[-1]
    total_gain = final["pass_rate"] - history[0]["pass_rate"]

    print(f"\n{'╔'+'═'*58+'╗'}")
    print(f"║{'ملخص التحسين النهائي':^58}║")
    print(f"{'╠'+'═'*58+'╣'}")
    print(f"║  الجولات المكتملة : {len(history)-1:<40}║")
    print(f"║  النسبة قبل        : {history[0]['pass_rate']:.1f}%{'':<37}║")
    print(f"║  النسبة بعد        : {final['pass_rate']:.1f}%{'':<37}║")
    print(f"║  الكسب الإجمالي   : {total_gain:+.1f}%{'':<37}║")
    print(f"{'╠'+'═'*58+'╣'}")
    if current_model != model_path:
        print(f"║  النموذج المحسّن: {current_model:<42}║")
        print(f"║  للمحادثة: python chat.py --model {current_model:<23}║")
    print(f"{'╚'+'═'*58+'╝'}\n")

    # حفظ تاريخ التحسين
    history_path = os.path.join(base_output, "improvement_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump({"rounds": history, "total_gain": total_gain}, f,
                  ensure_ascii=False, indent=2)
    print(f"  ✓ تاريخ التحسين محفوظ: {history_path}")


if __name__ == "__main__":
    main()
