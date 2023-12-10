#include <iostream>
#include <fstream>
#include <random>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <functional>
#include <typeinfo>
 
#include <vector>
#include <array>
#include <valarray>
#include <queue>
#include <stack>
#include <set>
#include <map>
 
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cassert>
#include <cmath>
#include <climits>
using namespace std;
 
#define int long long int
#define ll long long
#define vi vector<int>
#define vll vector<long long>
#define all(x) (x).begin(), (x).end()
#define rall(x) (x).rbegin(), (x).rend()
#define sz(x) (int)(x).size()
#define pb push_back
#define X first
#define Y second
#define fast ios_base::sync_with_stdio(false),cin.tie(0),cout.tie(0)
#define ln cout<<"\n"
#define pii pair<int,int>
#define deb(x) cout<<#x<<" = "<<x<<'\n'


struct D2SegTree {
	/// for sum
	int sz;
	int NETRAL = 0;
	
	struct SubTree {
		int sz;
		vector<int> tree;
		int NETRAL = 0;
		
		SubTree(int n) {
			init(n);
		}
		
		void init(int n) {
			sz = 1;
			while (sz < n) sz *= 2;
			tree.resize(sz * 2 - 1, NETRAL);
		}
		
		void build(vector<int> & v, int x, int lx, int rx) {
			if (lx == rx-1) {
				if (lx < (int)v.size()) {
					tree[x] = v[lx];
				} else {
					tree[x] = NETRAL;
				}
				return;
			}
			int m = (lx + rx) / 2;
			build(v, x*2+1, lx, m);
			build(v, x*2+2, m, rx);
			tree[x] = tree[x*2+1] + tree[x*2+2];
		}
		
		void build(vector<int > &v) {
			init(v.size());
			build(v, 0, 0, sz);
		}
		int get(int x, int lx, int rx, int l, int r) {
			if (r <= lx || rx <= l) return NETRAL;
			if (l <= lx && rx <= r) return tree[x];
			
			int m = (lx + rx) / 2;
			int L = get(x * 2 + 1, lx, m, l, r);
			int R = get(x * 2 + 2, m, rx, l, r);
			return L + R;
		}
		int get(int l, int r) {
			return get(0, 0, sz, l, r);
		}
		void update(int x, int lx, int rx, int i, int v) {
			if (rx == lx + 1) {
				tree[x] = v;
				return;
			}
			int m = (lx + rx) / 2;
			if (i < m) update(x * 2 + 1, lx, m, i, v);
			else update(x * 2 + 2, m, rx, i, v);
			tree[x] = tree[x * 2 + 1] + tree[x * 2 + 2];
		}
		void update(int i, int v) {
			update(0, 0, sz, i, v);
		}
	};

	vector<SubTree> tree;
	void init(int n, int m) {
		sz = 1;
		while (sz < n) sz *= 2;
		
		tree.resize(sz * 2 - 1, SubTree(m));
	}
	
	void build(vector<vector<int> > & v, int x, int lx, int rx) {
		if (lx == rx-1) {
			if (lx < (int)v.size()) {
				tree[x].build(v[lx]);
			}
			return;
		}
		int m = (lx + rx) / 2;
		build(v, x*2+1, lx, m);
		build(v, x*2+2, m, rx);
		vector<int> temp(v[0].size());
		for(int i=0; i<(int)v[0].size(); i++) {
			temp[i] = tree[x*2+1].get(i, i+1) + tree[x*2+2].get(i, i+1);
		}
		tree[x].build(temp);
	}
	
	void build(vector<vector<int> > &v) {
		init(v.size(), v[0].size());
		build(v, 0, 0, sz);
	}
	int get(int x, int lx, int rx, int l, int r, int t, int b) {
		if (r <= lx || rx <= l) return NETRAL;
		if (l <= lx && rx <= r) return tree[x].get(t, b);
		
		int m = (lx + rx) / 2;
		int L = get(x * 2 + 1, lx, m, l, r, t, b);
		int R = get(x * 2 + 2, m, rx, l, r, t, b);
		return L + R;
	}
	int get(int l, int r, int t, int b) {
		return get(0, 0, sz, l, r, t, b);
	}
	
	void update(int x, int lx, int rx, int i, int j, int v) {
		if (rx == lx + 1) {
			tree[x].update(j, v);
			return;
		}
		int m = (lx + rx) / 2;
		if (i < m) update(x * 2 + 1, lx, m, i, j, v);
		else update(x * 2 + 2, m, rx, i, j, v);
		tree[x].update(j, tree[x*2+1].get(j, j+1) + tree[x*2+2].get(j, j+1));
	}
	
	void update(int i, int j, int v) {
		update(0, 0, sz, i, j, v);
	}
};


struct KosarajuAlgorithm {
	// Kosaraju's Algorithm for finding Strongly Connected Components
    vector<vector<int> > G, GR;
    vector<int> order, comp, res;
    vector<bool> vis;
    int n, m;
    int clan;

	void build(vector<pair<int, int> > &v, int sz) {
		/// v is the edge list, sz is the number of nodes
		n = sz;
		m = v.size();
		G.assign(n+1, vector<int>());
		GR.assign(n+1, vector<int>());
		for(auto i: v) {
			G[i.first].push_back(i.second);
			GR[i.second].push_back(i.first);
		}
		vis.assign(n+1, 0);
		res.assign(n+1, 0);
		clan = 0;
		for(int i=1; i<=n; i++) if (!vis[i]) dfs1(i);
		reverse(order.begin(),order.end());
		vis.assign(n+1, 0);
		for(int i: order) {
			if (!vis[i]) {
				dfs2(i);
				clan ++;
				for(int v: comp) res[v] = clan;
				comp.clear();
			}
		}
	}

	void dfs1(int u) {
		vis[u] = 1;
		for(int v: G[u]) {
			if (!vis[v]) 
				dfs1(v);
		}
		order.push_back(u);
	}
	
	void dfs2(int u) {
		vis[u] = 1;
		comp.push_back(u);
		for(int v: GR[u]) {
			if (!vis[v]) {
				dfs2(v);
			}
		}
	}

	vector<vector<int> > get_clans() {
		vector<vector<int> > clans(clan+1);
		for(int i=1; i<=n; i++) {
			clans[res[i]].push_back(i);
		}
		return clans;
	}
};


namespace MathUtil {
	vector<int> get_phi(int n) {
		vector<int> phi(n+1);
		for (int i = 1; i <= n; i++) {
			for (int j = i; j <= n; j += i) {
				if (j > i) phi[j] -= phi[i];
			}
		}
		return phi;
	}

	vector<vector<int> > get_divs(int n) {
		vector<vector<int> > divs(n+1);
		for (int i = 1; i <= n; i++) {
			for (int j = i; j <= n; j += i) {
				divs[j].push_back(i);
			}
		}
		return divs;
	}

	vector<bool> get_prime_mask(int n) {
		vector<bool> prime(n+1, true);
		prime[0] = prime[1] = false;
		for (int i = 2; i <= n; i++) {
			if (prime[i]) {
				for (long long j = 1LL * i * i; j <= n; j += i) {
					prime[j] = false;
				}
			}
		}
		return prime;
	}
	
	long long BinPow(long long a, long long b, long long mod) {
		long long res = 1;
		while (b) {
			if (b & 1) res = (res * a) % mod;
			a = (a * a) % mod;
			b >>= 1;
		}
		return res;
	}

	// abs and devide it by 2
	long long area(pair<int, int> a, pair<int, int> b, pair<int, int> c) {
		return (b.second - a.second) * (c.first - b.first) - (b.first - a.first) * (c.second - b.second);
	}
};


struct TwoSAT {
    int n;
    vector<vector<int>> g, gr;
    vector<int> comp, topological_order, answer;
    vector<bool> vis;

    TwoSAT() {}

    TwoSAT(int _n) { init(_n); }

    void init(int _n) {
        n = _n;
        g.assign(2 * n, vector<int>());
        gr.assign(2 * n, vector<int>());
        comp.resize(2 * n);
        vis.resize(2 * n);
        answer.resize(2 * n);
    }

    void add_edge(int u, int v) {
        g[u].push_back(v);
        gr[v].push_back(u);
    }

    // At least one of them is true
    void add_clause_or(int i, bool f, int j, bool g) {
        add_edge(i + (f ? n : 0), j + (g ? 0 : n));
        add_edge(j + (g ? n : 0), i + (f ? 0 : n));
    }

    // Only one of them is true
    void add_clause_xor(int i, bool f, int j, bool g) {
        add_clause_or(i, f, j, g);
        add_clause_or(i, !f, j, !g);
    }

    // Both of them have the same value
    void add_clause_and(int i, bool f, int j, bool g) {
        add_clause_xor(i, !f, j, g);
    }

    void dfs(int u) {
        vis[u] = true;

        for (const auto &v : g[u])
            if (!vis[v]) dfs(v);

        topological_order.push_back(u);
    }

    void scc(int u, int id) {
        vis[u] = true;
        comp[u] = id;

        for (const auto &v : gr[u])
            if (!vis[v]) scc(v, id);
    }

	// main function
    bool satisfiable() {
        fill(vis.begin(), vis.end(), false);

        for (int i = 0; i < 2 * n; i++)
            if (!vis[i]) dfs(i);

        fill(vis.begin(), vis.end(), false);
        reverse(topological_order.begin(), topological_order.end());

        int id = 0;
        for (const auto &v : topological_order)
            if (!vis[v]) scc(v, id++);

        for (int i = 0; i < n; i++) {
            if (comp[i] == comp[i + n]) return false;
            answer[i] = (comp[i] > comp[i + n] ? 1 : 0);
        }

        return true;
    }
};


namespace GraphUtils {
	vector<int> topological_sort(vector<pair<int, int> > &v, int sz) {
		// v is the edge list, sz is the number of nodes
		vector<vector<int> > g(sz+1);
		vector<int> indeg(sz+1, 0);
		vector<int> order;
		for(auto i: v) {
			indeg[i.second] ++;
			g[i.first].push_back(i.second);
		}
		queue<int> q;
		for(int i=1; i<=sz; i++) {
			if (indeg[i] == 0) q.push(i), order.push_back(i);
		}
		while (!q.empty()) {
			int u = q.front();
			q.pop();
			for(auto i: g[u]) {
				indeg[i] --;
				if (indeg[i] == 0) q.push(i), order.push_back(i);
			}
		}
		return order;
	}

	vector<ll> disktra(vector<vector<int> > &v, int sz, int source) {
		vector<ll> dist(sz+1, (ll)1e18);
		dist[source] = 0;
		priority_queue<pair<ll, ll>, vector<pair<ll, ll> >, greater<pair<ll, ll> > > pq;
		pq.push({0, source});
		while (!pq.empty()) {
			auto [d, u] = pq.top();
			pq.pop();
			if (dist[u] != d) continue;
			for(auto v: v[u]) {
				if (dist[v] > dist[u] + 1) {
					dist[v] = dist[u] + 1;
					pq.push({dist[v], v});
				}
			}
		}
		return dist;
	}

	// pair<int, int> dfs(int v, int u = -1) {
	// 	// dfs for find max depth and the node from which the max depth is achieved && diameter of the tree  mx.first = dfs(1) && mx.second = dfs(mx.first)
	// 	if (g[v].size() == 0) return {0, v}; 
	// 	if (g[v].size() == 1 && u != -1) return {0, v};
	// 	pair<int, int> mx = {-1, -1};
	// 	for(int x: g[v]) {
	// 		if (x == u) continue;
	// 		pair<int, int> m = dfs(x, v);
	// 		if (mx < m) {
	// 			mx = m;
	// 		}
	// 	}
	// 	return {mx.first + 1, mx.second};
	// }
};

namespace Treap {
	/// cat and merge arrays fast log(n) for operation any length
	// Treap::node *A, *B, *C, *D;
    // Treap::split(treap, A, B, x - 1); 
    // Treap::split(B, D, C, y - x + 1);
    // Treap::merge(treap, A, C);  treap = A + C
    // Treap::merge(treap, treap, D);  treap = treap + D
	struct node {
	    node *left, *right;
	    int weight, size;
	    int value;
	    node(int v) {
	        left = right = NULL;
	        weight = rand();
	        size = 1;
	        value = v;
	    }
	};
	int size(node *treap) {
	    if (treap == NULL) return 0;
	    return treap->size;
	}
	void split(node *treap, node *&left, node *&right, int k) {
	    if (treap == NULL) {
	        left = right = NULL;
	    } 
	    else {
	        if (size(treap->left) < k) {
	            split(treap->right, treap->right, right, k - size(treap->left)-1);
	            left = treap;
	        } 
	        else {
	            split(treap->left, left, treap->left, k);
	            right = treap;
	        }
	        treap->size = size(treap->left) + size(treap->right) + 1;
	    }
	}
	
	void merge(node *&treap, node *left, node *right) {
	    if (left == NULL) treap = right;
	    else if(right == NULL) treap = left;
	    else {
	        if (left->weight < right->weight) {
	            merge(left->right, left->right, right);
	            treap = left;
	        } 
	        else {
	            merge(right->left, left, right->left);
	            treap = right;
	        }   
	        treap->size = size(treap->left) + size(treap->right) + 1;
	    }
	}
	void print(node *treap) {
	    if (treap == NULL) return;
	    print(treap->left);
	    cout << char(treap->value);
	    print(treap->right);
	}
};

struct DSU {
	int sz;
	vector<int> p;
	DSU (int n=0) {init(n);}
	void init(int n) {
		sz = n;
		p.resize(n+1);
		for(int i=0; i<=n; i++) p[i] = i;
	}
	int get(int x) {
		if (p[x] == x) return x;
		return p[x] = get(p[x]);
	}
	bool unite(int x, int y) {
		x = get(x);
		y = get(y);
		if (x == y) return false;
		p[x] = y;
		return true;
	}
};

struct SegmentTree {
	int sz;
	vector<int> tree;
	function<int(int, int)> combinator;
	int NETRAL = 0;
	SegmentTree(){};
	SegmentTree(function<int(int, int)> c, int netral=0) {
		combinator = c;
		NETRAL = netral;
	}
	
	void init(int n) {
		sz = 1;
		while (sz < n) sz *= 2;
		tree.resize(sz * 2 - 1, NETRAL);
	}
	
	void build(vector<int> & v, int x, int lx, int rx) {
		if (lx == rx-1) {
			if (lx < (int)v.size()) tree[x] = v[lx];
			else                    tree[x] = NETRAL;
			return;
		}
		int m = (lx + rx) / 2;
		build(v, x*2+1, lx, m);
		build(v, x*2+2, m, rx);
		tree[x] = combinator(tree[2*x+1], tree[x*2+2]);
	}
	
	void build(vector<int> &v) {
		init(v.size());
		build(v, 0, 0, sz);
	}
	int get(int x, int lx, int rx, int l, int r) {
		if (r <= lx || rx <= l) return NETRAL;
		if (l <= lx && rx <= r) return tree[x];
		int m = (lx + rx) / 2;
		int L = get(x * 2 + 1, lx, m, l, r);
		int R = get(x * 2 + 2, m, rx, l, r);
		return combinator(L, R);
	}
	int get(int l, int r) {return get(0, 0, sz, l, r);}
	void update(int x, int lx, int rx, int i, int v) {
		if (rx == lx + 1) {
			tree[x] = v;
			return;
		}
		int m = (lx + rx) / 2;
		if (i < m) update(x * 2 + 1, lx, m, i, v);
		else update(x * 2 + 2, m, rx, i, v);
		tree[x] = combinator(tree[2*x+1], tree[x*2+2]);
	}
	void update(int i, int v) {update(0, 0, sz, i, v);}
	int index_min(int x, int lx, int rx, int l, int r, int v) {
		if (r <= lx || rx <= l) return -1;
		if (l <= lx && rx <= r && tree[x] > v) return -1;
		if (lx == rx - 1) return lx;
		int m = (lx + rx) / 2;
		int L = index_min(x*2+1, lx, m, l, r, v);
		if (L != -1) return L;
		return index_min(x*2+2, m, rx, l, r, v);
	}
	int index_min(int l, int r) {
		/// min index in range
		int mn = get(l, r);
		return index_min(0, 0, sz, l, r, mn);
	}
};

// SegmentTree min_tree([](int x, int y) {return min(x, y);}, (int)(1e9+1));
// SegmentTree max_tree([](int x, int y) {return max(x, y);}, (int)(-1e9-1));
// SegmentTree sum_tree([](long long x, long long y) {return x + y;}, 0);

struct LCA {
	/// Lowest Common Ancestor
	vector<int> level, v, ind;
	SegmentTree tree{[](int x, int y) {return min(x, y);}, (int)1e9};
	int timer = 0;
	int sz;
	vector<vector<int> > G;
	LCA(){};
	LCA(vector<vector<int> > &g, int root=1) {build(g, root);}
	void build(vector<vector<int> > &g, int root=1){
		sz = g.size();
		G = g;
		level.resize(sz);
		ind.resize(sz);
		dfs(root);
		G.clear();
		vector<int> vals;
		for(int i=0; i<v.size(); i++) vals.push_back(level[v[i]]);
		tree.build(vals);
		vals.clear();
	}
	void dfs(int u, int p=-1, int d=0) {
		ind[u] = v.size();
		v.push_back(u);
		level[u] = d;
		for(int x: G[u]) if (x != p) dfs(x, u, d+1), v.push_back(u);
		
	}
	int answer(int u, int w) {
		int l = ind[u];
		int r = ind[w];
		if (r < l) swap(l, r);
		return v[tree.index_min(l, r+1)];
	}
};

struct HLD {
	/// HLD for max in (u, v) path for tree only
	vector<SegmentTree> trees;
	vector<vector<int> > clans;
	int sz;
	LCA lca;
	vector<int> top, level, subtrees, parent;
	
	void build(int n, vector<vector<int> > &g, vector<int> &vals) {
		sz = n;
		LCA lca1(g, 1);
		lca = lca1;
		top.assign(sz+1, 0);
		level.assign(sz+1, 0);
		subtrees.assign(sz+1, 0);
		parent.assign(sz+1, 0);
		trees.resize(sz+1);
		clans.resize(sz+1);
		
		dfs(g, 1);
		feat_clans(g, 1);
		vector<bool> f(n+1);
		for(int i=1; i<=n; i++) {
			int t = top[i];
			if (!f[t]) {
				trees[t].NETRAL = -1e9-1;
				trees[t].combinator = [](int x, int y){return max(x, y);};
				vector<int> vv(clans[t].size());
				for(int j=0; j<clans[t].size(); j++) {
					vv[j] = vals[clans[t][j]];
				}
				trees[t].build(vv);
				f[t] = true;			
			}
		}
		
		clans.clear();
		subtrees.clear();
	}
	
	
	void dfs(vector<vector<int> > &g, int u, int p=-1, int d=0) {
		parent[u] = p;
		level[u] = d;
		top[u] = u;
		subtrees[u] = 1;
		int mx = -1;
		for(int x: g[u]) {
			if (x != p) {
				dfs(g, x, u, d+1);
				if (mx == -1) mx = x;
				if (subtrees[mx] < subtrees[x]) {
					mx = x;
				}
				subtrees[u] += subtrees[x];
			}
		}
		if (mx != -1)
			top[mx] = u;
		else top[u] = u;
	}
	void feat_clans(vector<vector<int> > &g, int u, int p=-1) {
		clans[top[u]].push_back(u);
		for(int x: g[u]) {
			if (p != x) {
				if (top[x] == u) {
					top[x] = top[u];
				}
				feat_clans(g, x, u);
			}
		}
	}
	
	int get_upto_lca(int x, int y) {
		int res = -1e9-1;
		while (true) {
			int t = top[x];
			if (level[t] > level[top[y]]) {
				res = max(res, trees[t].get(0, level[x] - level[t] + 1));
				x = parent[t];
			} else {
				res = max(res, trees[t].get(level[y]-level[t], level[x] - level[t] + 1));
				return res;
			}
		}
		return 0LL;
	};
	
	int answer(int x, int y) {
		int l = lca.answer(x, y);
		return max(get_upto_lca(x, l), get_upto_lca(y, l));
	}
	
	void update(int s, int x) {
		trees[top[s]].update(level[s]-level[top[s]], x);
	}
};


vector<int> z_function(string &s) {
	int n = s.size();
	vector<int> z(n);
	for(int i=1, l=0, r=0; i<n; i++) {
		if (i <= r) z[i] = min(z[i-l], r-i+1);
		while (i + z[i] < n && s[z[i]] == s[i+z[i]]) z[i] ++;
		if (i + z[i] - 1 > r) l = i, r = i + z[i] - 1;
	}
	return z;
}

// Geometry
struct P {
	int x, y;
	void read() {cin >> x >> y;}
	P operator - (const P& p) const {return P{x - p.x, y - p.y};}
	P operator + (const P& p) const {return P{x + p.x, y + p.y};}
	void operator -= (const P& p) {x -= p.x; y -= p.y;}
	void operator += (const P& p) {x += p.x; y += p.y;}
	long long operator * (const P& p) const {return (long long) x * p.y - (long long) y * p.x;}
	long long treangle(const P& b, const P& c) const {return (b - *this) * (c - *this);}
};


ll BinPow(ll a, ll b, ll m) {
	ll ans = 1;

	while(b) {
		if (b & 1)
			ans = (ans * a) % m;
		a = a * a % m;
		b >>= 1;
	}
	return ans;
}

bool IsPrime(ll n) {
	if (n == 1) return false;
	ll x;

	for(int i = 0; i < 100; i ++) {
		x = rand() + 10;
		if (x % n == 0) x ++;
		if (BinPow(x, n-1, n) != 1) return false;
	}
	return true;
}

ll Phi(ll n) {
	if (IsPrime(n)) return n - 1;
	ll result = n;
	for (int i = 2; i * i <= n; i ++)
		if (n % i == 0) {
			while (n % i == 0)
			  n /= i;
			result -= result / i;
		}
	if (n > 1)
		result -= result / n;
	return result;
}

ll InverseMod(ll a, ll b, ll m, ll phi = -1) {
	if(phi == -1) phi = Phi(m);
	
	return a * BinPow(b, phi - 1, m) % m;
}
 
namespace factorization {
	ll modmul(ll x, ll y, ll p) {
		ll q = (__int128) x * y / p;
		ll result = (ll) ((ll)(x) * y - q * p) % p;
		
		return result < 0 ? result + p : result;
	}
	
  ll bgcd(ll x, ll y) {
    if (!x || !y) return x + y;
    int shift = __builtin_ctzll(x | y);
    x >>= __builtin_ctzll(x);
    do {
      y >>= __builtin_ctzll(y);
      if (x > y) swap(x, y);
      y -= x;
    } while (y);
    return x << shift;
  }
 
  ll pw(ll a, ll n, ll p) {
    ll res = 1;
    while (n) {
      if (n & 1) res = modmul(res, a, p);
      a = modmul(a, a, p);
      n >>= 1;
    }
    return res;
  }
 
  bool check_composite(ll n, int s, ll d, ll a) {
    ll x = pw(a, d, n);
    if (x == 1 || x == n - 1) return false;
    for (int it = 1; it < s; ++it) {
      x = modmul(x, x, n);
      if (x == n - 1) return false;
    }
    return true;
  }
 
  bool is_prime(ll n) {
    if (n < 4) return n > 1;
    int s = 0;
    ll d = n - 1;
    while (!(d & 1)) {
      d >>= 1;
      ++s;
    }
    static vector<ll> primes32{2, 7, 61};
    static vector<ll> primes64{2, 325, 9375, 28178, 450775, 9780504,
                                1795265022};
    static ll const BOUND = (ll)(4759123141ll);
    for (ll a : (n <= BOUND ? primes32 : primes64)) {
      if (n == a) return true;
      if (check_composite(n, s, d, a)) return false;
    }
    return true;
  }
 
  ll find_divisor(ll n, int c = 2) {
    auto f = [&](ll x) {
      auto r = modmul(x, x, n) + c;
      if (r >= n) r -= n;
      return r;
    };
    ll x = c;
    ll g = 1;
    ll q = 1;
    ll xs, y;
 
    int m = 128;
    int l = 1;
    while (g == 1) {
      y = x;
      for (int i = 1; i < l; ++i) {
        x = f(x);
      }
      int k = 0;
      while (k < l && g == 1) {
        xs = x;
        for (int i = 0; i < m && i < l - k; ++i) {
          x = f(x);
          q = modmul(q, llabs(y - x), n);
        }
        g = bgcd(q, n);
        k += m;
      }
      l *= 2;
    }
    if (g == n) {
      do {
        xs = f(xs);
        g = bgcd(llabs(xs - y), n);
      } while (g == 1);
    }
    return g == n ? find_divisor(n, c + 1) : g;
  }
 
  vector<pair<ll, int>> factorize(ll m) {
    if (m == 1) {
      return {};
    }
    vector<ll> fac;
    auto rec = [&fac](auto&& rec, ll m) -> void {
      if (is_prime(m)) {
        fac.push_back(m);
        return;
      }
      auto d = m % 2 == 0 ? 2 : find_divisor(m);
      rec(rec, d);
      rec(rec, m / d);
    };
    rec(rec, m);
    sort(fac.begin(), fac.end());
    vector<pair<ll, int>> ans;
    for (auto x : fac) {
      if (ans.empty() || ans.back().first != x) {
        ans.emplace_back(x, 0);
      }
      ++ans.back().second;
    }
    return ans;
  }
}
using factorization::factorize;
using factorization::is_prime;

const int N = 1e5 + 5, MOD = 1e9 + 7;

void t_main() {
}

signed main()
{
	fast;
	int t = 1;
	// cin >> t;
	while (t--) {
		t_main();	
	}
	return 0;
}
