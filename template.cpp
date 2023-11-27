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
 

struct KosarajuAlgorithm {
	// Kosaraju's Algorithm for finding Strongly Connected Components
    vector<vector<int> > G, GR;
    vector<int> order, comp, res;
    vector<bool> vis;
    int n, m;
    int clan;

	void build(vector<pair<int, int> > &v, int sz) {
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


struct MathUtil {
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
};

void t_main() {

}

signed main()
{
	ios_base::sync_with_stdio(false);
	cin.tie(0);
	cout.tie(0);
	int t = 1;
	cin >> t;
	while (t--) {
		t_main();	
	}
	return 0;
}
