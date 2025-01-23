from pyvis.network import Network
import IPython
import matplotlib.pyplot as plt
import networkx
# function to plot the knowledge graph

### help link = https://gist.github.com/quadrismegistus/92a7fba479fc1e7d2661909d19d4ae7e
def plot_graph(NxGr):
  net = Network(notebook=True, cdn_resources='remote')#G = nx.DiGraph()
  nodes = list(NxGr.nodes)

  for node,node_attrs in NxGr.nodes(data=True):
        net.add_node(str(node),**node_attrs)
  for source,target,edge_attrs in NxGr.edges(data=True):

    net.add_edge(str(source),str(target),label = str(edge_attrs['relation']))


  plt.figure(figsize=(10,6) , dpi=300)
  # pos = nx.spring_layout(G, k=3, seed=1)

  # net.show(f"distance_{node_distance}_spring_length_{spring_length}.html")

  # net.show_buttons(filter_=['KG'])
  net.show("KG.html")

  # fig = IPython.display.HTML(filename='KG.html')
  # return fig