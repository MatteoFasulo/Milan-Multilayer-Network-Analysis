import networkx as nx
import osmnx as ox

import numpy as np

import matplotlib.colors as mcolors
import folium
import folium.plugins

from osmgraph import OSMGraph

class MultilayerNetwork(OSMGraph):
    def __init__(self):
        self.layers = {}
        self.G = nx.Graph()
        self.SEED = 42

    def add_layer(self, layer_name: str, layer_graph: nx.Graph):

        if layer_name in self.layers.keys():
            print(f"ERROR: Layer {layer_name} already exists, skipping...")
        else:
            self.layers[layer_name] = layer_graph
            try:
                self.G = nx.union(G=self.G, H=layer_graph)
            except nx.NetworkXError:
                common_nodes = set(self.G.nodes).intersection(set(layer_graph.nodes))
                # Rename the common nodes in the layer graph using the maximum id + 1
                max_id = max(self.G.nodes) + 1
                mapping = {node: max_id+i for i, node in enumerate(common_nodes)}
                layer_graph = nx.relabel_nodes(layer_graph, mapping)
                self.G = nx.union(G=self.G, H=layer_graph) 

            print(f"Layer {layer_name} added.")

    def spatial_join(self, layer1: str, layer2: str, distance_threshold: int = 100):
        transfer_layer_name = f'{layer1}--{layer2}'

        self.layers[transfer_layer_name] = nx.Graph()

        layer1_copy = self.layers[layer1].copy()
        layer2_copy = self.layers[layer2].copy()

        edges_added = 0
        for n in layer1_copy.nodes:
            for m in layer2_copy.nodes:
                distance_mt = ox.distance.great_circle(layer1_copy.nodes[n]['y'], layer1_copy.nodes[n]['x'], layer2_copy.nodes[m]['y'], layer2_copy.nodes[m]['x'])
                if distance_mt <= distance_threshold:
                    self.G.add_edge(n, m, 
                        layer_name=transfer_layer_name,
                        weight=distance_mt,
                        length=distance_mt)
                    edges_added += 1
        
        print(f'Added {edges_added} transfer edges between {layer1} and {layer2}.')

    def summary(self):
        layers = {layer: (len([n for n,attrdict in self.G.nodes.items() if attrdict.get('layer_name') == layer]), 
                          len([(u,v,d) for u,v,d in self.G.edges(data=True) if d['layer_name'] == layer])) for layer in self.layers.keys()} 

        print ('{0: <16}'.format('layer') + '\tnodes \tedges')
        print ('-'*40)
        for layer in layers:
            print ('{0: <16}'.format(layer), "\t", layers[layer][0], "\t", layers[layer][1])

    def compare_apl(self, layer_name: str, verbose: bool = False):
        # Get the isolated layer graph
        layer_graph = self.layers[layer_name]

        # Compute APL of the isolated layer
        if nx.is_connected(layer_graph):
            layer_apl = nx.average_shortest_path_length(layer_graph, weight='length')
        else:
            max_connected_component = max(nx.connected_components(layer_graph), key=len)
            layer_apl = nx.average_shortest_path_length(layer_graph.subgraph(max_connected_component), weight='length')

        # Initialize variables to compute multilayer APL for specified layer paths
        total_length = 0
        num_pairs = 0

        # Iterate over all nodes in the layer graph
        for source in layer_graph.nodes:
            # Compute shortest paths from the source in the multilayer graph
            paths_from_source = nx.single_source_dijkstra_path_length(self.G, source, weight='length')

            # Consider only paths that end within the same layer
            for target, path_length in paths_from_source.items():
                if target in layer_graph.nodes and path_length > 0:
                    if verbose:
                        print(f"Source: {source}, Target: {target}, Length: {path_length}")
                    total_length += path_length
                    num_pairs += 1

        # Compute multilayer APL
        try:
            multilayer_apl = total_length / num_pairs
        except ZeroDivisionError:
            multilayer_apl = 0
            print(f"No valid paths found for layer {layer_name} in the multilayer graph.")

        return layer_apl, multilayer_apl

    def plot_map(self):
        # Extract nodes of the first layer just to center the map
        tmp_nodes, _ = self.get_gdf(self.layers_graphs[self.layers[0]])

        # Create a folium map
        m = folium.Map(
            tiles="CartoDB positron", 
            location=[tmp_nodes['y'].mean(), tmp_nodes['x'].mean()], 
            zoom_start=12
        )

        fg = folium.FeatureGroup(control=False)
        m.add_child(fg)

        # Add a child to the feature group for each layer
        for layer_name, layer_graph in self.layers_graphs.items():
            # Get the GeoDataFrame of the layer
            nodes, edges = self.get_gdf(layer_graph)
            # Create a feature group for the current layer
            subgroup = folium.plugins.FeatureGroupSubGroup(fg, layer_name.capitalize())
            m.add_child(subgroup)

            # Get a random color for the layer
            subgroup_color = np.random.choice(list(mcolors.CSS4_COLORS.values()))

            # Add the edges to the map
            for i, row in edges.iterrows():
                line = folium.PolyLine(
                    locations=[(row['geometry'].coords[i][1], row['geometry'].coords[i][0]) for i in range(len(row['geometry'].coords))],
                    color=row.get('colour', subgroup_color),
                    weight=5,
                    tooltip=row['ref']
                )

                subgroup.add_child(line)

            # Add the nodes to the map
            for _, node_row in nodes.iterrows():
                circle = folium.Circle(
                    location=[node_row['y'], node_row['x']],
                    radius=30,
                    color='black',
                    fill=True,
                    fill_color=node_row.get('colour', 'black'),
                    tooltip=node_row['name']
                )
                subgroup.add_child(circle)

        # Add the cross-layer to the map
        

        folium.plugins.LocateControl().add_to(m)
        folium.plugins.MiniMap(position='bottomright').add_to(m)
        folium.plugins.MeasureControl(position='bottomleft', primary_length_unit='kilometers', secondary_length_unit='meters').add_to(m)

        formatter = "function(num) {return L.Util.formatNum(num, 3) + ' º ';};"
        folium.plugins.MousePosition(
            position="topright",
            separator=" | ",
            empty_string="NaN",
            lng_first=True,
            num_digits=20,
            prefix="Coordinates:",
            lat_formatter=formatter,
            lng_formatter=formatter,
        ).add_to(m)

        folium.LayerControl().add_to(m)

        return m 