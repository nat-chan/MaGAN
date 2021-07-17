# %%
import math
from heapq import heappush, heappop
import numpy as np
from scipy.sparse.csgraph import shortest_path, floyd_warshall, dijkstra, bellman_ford, johnson
from scipy.sparse import csr_matrix
# %%
def solve(a, b, x, y):
    if b-a>=0:
        return min((b-a)*y+x, (b-a)*2*x+x)
    else:
        return min((a-b)*y+x, (a-b)*2*x-x)

# %%
def solve2(a, b, x, y):
    l = np.full((200,200), np.inf)
    for i in range(100):
        l[i,i+100] = x
        l[i+100,i] = x
    for i in range(99):
        l[i+1,i+100] = x
        l[i+100,i+1] = x
    for i in range(99):
        l[i,i+1] = y
        l[i+1,i] = y
        l[i+100,i+101] = y
        l[i+101,i+100] = y
    return int(shortest_path(l)[a-1, b+100-1])

def solve3(a, b, x, y):
    adj = [[] for _ in range(200)]
    for i in range(100):
        adj[i].append((i+100, x))
        adj[i+100].append((i, x))
    for i in range(99):
        adj[i+1].append((i+100, x))
        adj[i+100].append((i+1, x))
    for i in range(99):
        adj[i].append((i+1, y))
        adj[i+1].append((i, y))
        adj[i+100].append((i+101, y))
        adj[i+101].append((i+100, y))
    d = dijkstra(a-1, 200, adj)
    return d[b+100-1]

def dijkstra(s, n, adj): # (始点, ノード数)
    INF = float('inf')
    dist = [INF] * n
    hq = [(0, s)] # (distance, node)
    dist[s] = 0
    seen = [False] * n # ノードが確定済みかどうか
    while hq:
        v = heappop(hq)[1] # ノードを pop する
        seen[v] = True
        for to, cost in adj[v]: # ノード v に隣接しているノードに対して
            if seen[to] == False and dist[v] + cost < dist[to]:
                dist[to] = dist[v] + cost
                heappush(hq, (dist[to], to))
    return dist

# %%

x = 2
y = 2*x - 1
a = 100
# %%
for b in range(1, 100+1):
    if solve(a,b,x,y) != solve2(a,b,x,y):
        print("###########################")
    print(a,b,x,y, solve(a,b,x,y) == solve2(a,b,x,y))
# %%
a = 100
b = 98
x = 2
y = 3
print(solve(a, b, x, y))
print(solve2(a, b, x, y))
print(solve3(a, b, x, y))
# %%
from dataclasses import dataclass
from typing import Namedtuple

@dataclass
class PointMutable:
    x: int
    y: int

class PointImmutable(Namedtuple):
    x: int
    y: int

def f(p):
    p.x = 1
    p.y = 1

p_mutable = PointMutable(x=0, y=0)
print('before', p_mutable)
f(p_mutable)
print('after', p_mutable)

# %%
