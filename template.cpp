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
 

const int N = 3e5;
vector<int> G[N], GR[N], comp, res(N);
vector<int> order;
int n, m, vis[N];
int clan = 0;

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


 
signed main()
{
    cin >> n >> m;
	vector<pair<int, int> > v;
	for(int i=0; i<m; i++) {
		int a, b;
		cin >> a >> b;
		v.push_back({a, b});
	}
	KosarajuAlgorithm kosa;
	kosa.build(v, n);
	cout << kosa.clan << endl;
	for(int i=1; i<=n; i++) cout << kosa.res[i] << " ";
    return 0;
}
