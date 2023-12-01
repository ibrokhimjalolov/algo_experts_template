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

struct SegTree {
	int sz;
	vector<int> tree;
	int NETRAL = 0;
	
	SegTree(int n) {
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

void t_main() {

}

signed main()
{
	fast;
	int t = 1;
	cin >> t;
	while (t--) {
		t_main();	
	}
	return 0;
}
