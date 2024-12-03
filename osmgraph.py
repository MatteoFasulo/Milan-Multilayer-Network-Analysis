# Description: This module contains a class that can be used to create a graph of public transportation routes from OpenStreetMap data.
import requests
import json
from collections import Counter
from itertools import combinations
from typing import List, Tuple
from pathlib import Path

from tqdm import tqdm

import networkx as nx
import osmnx as ox
import geopandas

import numpy as np

import branca.colormap as cm
import matplotlib.pyplot as plt
import folium
import folium.plugins

from sklearn.cluster import DBSCAN
import powerlaw

class OSMGraph:
    def __init__(self, location: str, kind: str, timeout: int = 30):
        assert isinstance(location, str), 'Location must be a string'
        self.location = location

        if kind not in ['bus', 'tram', 'subway']:
            raise ValueError('Invalid kind of transport, must be bus, tram or subway')

        self.kind = kind

        assert isinstance(timeout, int), 'Timeout must be an integer'
        self.timeout = timeout

        self.cache_dir = Path('cache')
        self.cache_dir.mkdir(exist_ok=True)
        self.img_dir = Path('images')
        self.img_dir.mkdir(exist_ok=True)
        self.SEED = 42
        self.crs = 'EPSG:4326'

    def get_query(self) -> str:
        """
        Generates an Overpass QL query based on the type of public transportation.

        Returns:
            str: The Overpass QL query string.

        Raises:
            ValueError: If the kind of transportation is not recognized.

        The query is generated based on the following types of transportation:
        - 'bus': Queries for bus routes in the specified location.
        - 'tram': Queries for tram routes in the specified location.
        - 'subway': Queries for subway routes in the specified location.

        The query includes a timeout parameter and searches within the specified location area.
        """
        if self.kind == 'bus':
            return f'''
            [out:json]
            [timeout:{self.timeout}];
            area["name:en"="{self.location}"]->.searchArea;
            relation["type"="route"]["route"="bus"]["operator"="ATM"](area.searchArea);
            out meta;
            >;
            out body;
            '''
        elif self.kind == 'tram':
            return f'''
            [out:json]
            [timeout:{self.timeout}];
            area["name:en"="{self.location}"]->.searchArea;
            relation["route"="tram"](area.searchArea);
            out meta;
            >;
            out body;
            '''
        elif self.kind == 'subway':
            return f'''
            [out:json]
            [timeout:{self.timeout}];
            area["name:en"="{self.location}"]->.searchArea;
            relation["route"="subway"](area.searchArea);
            out meta;
            >;
            out body;
            '''

    def get_data(self, save_cache: bool = True) -> dict:
        """
        Fetches data from the Overpass API and caches the result.

        This method first checks if the data for the current query is already
        cached. If it is, the cached data is loaded and returned. If not, a
        request is made to the Overpass API to fetch the data. The fetched data
        is then optionally cached and returned.

        Args:
            save_cache (bool): If True, the fetched data will be saved to the
                       cache. Defaults to True.

        Returns:
            dict: The data fetched from the Overpass API, either from the cache
              or from the API directly.

        Raises:
            requests.exceptions.HTTPError: If the request to the Overpass API
                           fails.
        """
        # Check that the query hash is not already in the cache
        query_hash = hash(self.get_query())
        cache_file = self.cache_dir / f'{query_hash}.json'
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)

        # If not, make the request
        overpass_url = "https://overpass-api.de/api/interpreter"
        response = requests.get(overpass_url, params={'data': self.get_query()})
        # Check if the request was successful
        if response.status_code == 200:
            if save_cache:
                # Save the response in cache folder
                query_hash = hash(self.get_query())
                cache_file = self.cache_dir / f'{query_hash}.json'
                with open(cache_file, 'w') as f:
                    json.dump(response.json(), f, indent=4)
                return response.json()
        # If the request failed, raise an exception
        else:
            raise response.raise_for_status()

    def get_route_elements(self, osm_data: dict) -> List[dict]:
        """
        Extracts route elements from OpenStreetMap (OSM) data.

        This method filters the elements in the provided OSM data to return only those
        that have a 'tags' key and contain a 'route' key within the 'tags' dictionary.

        Args:
            osm_data (dict): A dictionary containing OSM data, which includes a list of elements.

        Returns:
            list: A list of elements that have a 'tags' key and contain a 'route' key within the 'tags' dictionary.
        """
        return [element for element in osm_data['elements'] if 'tags' in element and 'route' in element['tags']]

    def get_node_elements(self, osm_data: dict) -> dict:
        """
        Extracts and returns node elements from OpenStreetMap (OSM) data.

        Args:
            osm_data (dict): A dictionary containing OSM data, where 'elements' is a key 
                             that holds a list of elements, each being a dictionary with 
                             various attributes including 'id' and 'type'.

        Returns:
            dict: A dictionary where the keys are the IDs of the node elements and the 
                  values are the corresponding element dictionaries.
        """
        return {element['id']: element for element in osm_data['elements'] if element['type'] == 'node'}

    def get_graph(self, simplify: bool = False, *args, **kwargs) -> nx.Graph:
        """
        Generate a graph representation of the OpenStreetMap (OSM) data.
        Parameters:
        simplify (bool): If True, the graph will be simplified using the simplify_graph method. Default is False.
        *args: Additional positional arguments to pass to the simplify_graph method.
        **kwargs: Additional keyword arguments to pass to the simplify_graph method.
        Returns:
        networkx.Graph: A NetworkX graph object representing the OSM data.
        The method performs the following steps:
        1. Instantiates a Graph.
        2. Retrieves OSM data using the get_data method.
        3. Extracts route elements and node elements from the OSM data.
        4. Adds nodes and edges to the graph based on the route and node elements.
        5. Sets the graph's coordinate reference system (CRS) to 'EPSG:3857'.
        6. Optionally simplifies the graph if the simplify parameter is True.
        7. Stores the generated graph in the instance variable self.G and returns it.
        """
        # Instantiate a Graph
        G = nx.Graph()

        # Get the data
        osm_data = self.get_data()

        # Get the routes
        route_elements = self.get_route_elements(osm_data)
        # Get the nodes
        node_elements = self.get_node_elements(osm_data)

        # Add nodes and edges
        for route in route_elements:
            stop_nodes = [member for member in route['members'] if member['role'] == 'stop']

            # Add nodes
            for node in stop_nodes:
                ref = node['ref']
                if ref in node_elements:
                    node_data = node_elements[ref]
                    node_tags = node_data.get('tags', {})
                    name = node_tags.get('name', str(ref))
                    line = route['tags'].get('ref', 'Unknown')

                    G.add_node(node_for_adding=ref, id=node_data['id'], osmid=node_data['id'], pos=(node_data['lon'], node_data['lat']), x=node_data['lon'], y=node_data['lat'], name=name, line=line, layer_name=self.kind)

            # Add edges between consecutive stop nodes
            for i in range(len(stop_nodes) - 1):
                if stop_nodes[i]['ref'] in node_elements and stop_nodes[i+1]['ref'] in node_elements: #should avoid to add nodes with no data
                    G.add_edge(u_of_edge=stop_nodes[i]['ref'], v_of_edge=stop_nodes[i+1]['ref'], **route['tags'], layer_name=self.kind)

        G.graph = {'crs': self.crs} # TODO: Check if this is the appropriate CRS

        if simplify:
            G = self.simplify_graph(G, *args, **kwargs)
            
        self.G = G

        # Add edge length attribute
        G = self.add_edge_lengths(G)
        
        self.G = G
        return G

    def add_edge_lengths(self, G: nx.Graph) -> nx.Graph:
        nodes, edges = self.get_gdf()
        G = ox.graph_from_gdfs(nodes, edges) # returns a MultiDiGraph
        G = ox.distance.add_edge_lengths(G)
        spn = ox.stats.count_streets_per_node(G)
        nx.set_node_attributes(G, values=spn, name="street_count")
        edges_length = nx.get_edge_attributes(G, "length")
        nx.set_edge_attributes(G, values=edges_length, name="weight")

        # Convert back to networkx.Graph
        if isinstance(G, nx.MultiDiGraph):
            print("Converting to Graph")
            G = nx.Graph(G)
        
        return G

    def centroid(self, positions: list[tuple]) -> Tuple[float, float]:
        """
        Calculate the centroid (average latitude and longitude) of a list of positions.

        Args:
            positions (list[tuple]): A list of tuples where each tuple contains two floats 
                                     representing the latitude and longitude of a position.

        Returns:
            tuple: A tuple containing two floats representing the average latitude and longitude.
        """
        avg_lat = sum([x[0] for x in positions]) / len(positions)
        avg_lon = sum([x[1] for x in positions]) / len(positions)

        return avg_lat, avg_lon

    def ___simplify_graph(self, G, distance_threshold: int = 50, verbose: bool = False):
        """
        Deprecated method for simplifying a graph by merging nodes that are close to each other.
        """
        node_to_be_merged = {k[1]['osmid']: [] for k in list(G.nodes(data=True))} # {ID: [node2_ID, node3_ID]}
        for node in tqdm(G.nodes(data=True), desc='Finding nodes to merge', disable=not verbose):
            node1 = node[1]
            for node2 in G.nodes(data=True):
                node2 = node2[1]
                different_nodes = node1['osmid'] != node2['osmid']
                same_name = node1['name'] == node2['name']
                spatially_close = ox.distance.great_circle(node1['y'], node1['x'], node2['y'], node2['x']) <= distance_threshold

                if different_nodes and (same_name or spatially_close):
                        node_to_be_merged[node1['osmid']].append(node2['osmid'])

        #visited = set()
        #for key in list(node_to_be_merged.keys()):
        #    if key not in visited and len(node_to_be_merged[key]) > 0: 
        #        for node in node_to_be_merged[key]:
        #            if node in node_to_be_merged and node not in visited:
        #                node_to_be_merged[key].extend(node_to_be_merged[node])
        #                visited.add(node)

        #print(node_to_be_merged)

        # Merge nodes
        count = 0
        for node_id, nodes_to_merge in tqdm(node_to_be_merged.items(), desc='Merging nodes', disable=not verbose):
            if len(nodes_to_merge) == 0:
                continue

            if node_id < min(nodes_to_merge):

                centroid_position = self.centroid([G.nodes[node_id]['pos']] + [G.nodes[node]['pos'] for node in nodes_to_merge])
                for node in nodes_to_merge:
                    G = nx.contracted_nodes(G, node_id, node, self_loops=False)
                G.nodes[node_id]['pos'] = centroid_position
                G.nodes[node_id]['x'] = centroid_position[0]
                G.nodes[node_id]['y'] = centroid_position[1]
                count += 1
                
        if verbose:
            print(f'Merged {count} nodes.')
        return G

    def simplify_graph(self, G: nx.Graph, distance_threshold: int = 50, verbose: bool = False) -> nx.Graph:
        """
        Simplifies the given graph by merging nodes that are within a specified distance threshold.
        Parameters:
        G (networkx.Graph): The input graph to be simplified.
        distance_threshold (int, optional): The distance threshold in meters for merging nodes. Default is 50.
        verbose (bool, optional): If True, prints additional information during processing. Default is False.
        Returns:
        networkx.Graph: The simplified graph with merged nodes.
        Notes:
        - The function uses DBSCAN clustering to group nodes that are within the distance threshold.
        - Nodes within the same cluster are merged into a single node, with the position updated to the centroid of the cluster.
        - The primary node for each cluster is chosen as the node with the smallest ID.
        - The position of the primary node is updated to the centroid of the cluster.
        """
        # Extract positions and corresponding node IDs
        positions = np.array([G.nodes[node]['pos'] for node in G.nodes()])
        node_ids = list(G.nodes())
        
        # Perform DBSCAN clustering
        db = DBSCAN(eps=distance_threshold / 1e5, min_samples=1).fit(positions)  # Convert meters to kilometers if needed

        # Create a mapping from cluster label to node IDs
        cluster_map = {}
        for idx, label in enumerate(db.labels_):
            if label not in cluster_map:
                cluster_map[label] = []
            cluster_map[label].append(node_ids[idx])
        
        print("Found", len(cluster_map), "clusters.")
        # Merge nodes based on clusters
        count = 0
        for cluster_nodes in tqdm(cluster_map.values(), desc='Merging nodes', disable=not verbose):
            if len(cluster_nodes) < 2:
                continue
            
            centroid_position = self.centroid([G.nodes[node]['pos'] for node in cluster_nodes])
            primary_node = min(cluster_nodes)  # Choose one node as primary
            
            for node in cluster_nodes:
                if node != primary_node:
                    G = nx.contracted_nodes(G, primary_node, node, self_loops=False)
            
            # Update position of primary node to centroid position
            G.nodes[primary_node]['pos'] = centroid_position
            G.nodes[primary_node]['x'], G.nodes[primary_node]['y'] = centroid_position
            
            count += 1

        if verbose:
            print(f'Merged {count} clusters into centroids.')
        
        return G

        
    def visualize_graph(self) -> None:
        """
        Visualizes the graph using matplotlib and networkx.

        This method creates a plot of the graph with nodes and edges. Nodes are 
        colored based on their 'colour' attribute, labeled with their 'name' 
        attribute, and positioned according to their 'pos' attribute. Edges are 
        drawn in gray with some transparency.

        The plot is displayed with a title that includes the location and kind 
        of the graph.

        Returns:
            None
        """
        plt.figure(figsize=(15, 8))

        pos = nx.get_node_attributes(self.G, 'pos')
        node_labels = nx.get_node_attributes(self.G, 'name')
        node_colors = list(nx.get_node_attributes(self.G, 'colour').values())
        nx.draw(self.G, pos, with_labels=True, labels=node_labels, node_color=node_colors, node_size=100, font_size=2, font_color='white')
        nx.draw_networkx_edges(self.G, pos, edge_color='gray', alpha=0.7)
        plt.title(f"{self.location.title()} {self.kind.title()}")
        plt.show()

    def get_gdf(self, *args, **kwargs) -> geopandas.GeoDataFrame:
        """
        Convert the graph to GeoDataFrames.

        This method converts the graph stored in the instance to GeoDataFrames
        using the osmnx library's `graph_to_gdfs` function. If the graph is not
        already a MultiDiGraph, it will be converted to one.

        Parameters:
        *args : tuple
            Positional arguments to pass to the `graph_to_gdfs` function.
        **kwargs : dict
            Keyword arguments to pass to the `graph_to_gdfs` function.

        Returns:
        tuple
            A tuple containing two GeoDataFrames: one for the nodes and one for the edges.
        """
        if not isinstance(self.G, nx.MultiDiGraph):
            G = nx.MultiDiGraph(self.G)
        else:
            G = self.G
        return ox.graph_to_gdfs(G, *args, **kwargs)

    def delete_double_edges(self, copy: bool = True) -> nx.Graph:
        """
        Remove double edges from the graph.

        This method removes edges that appear more than once between any two nodes
        in the graph. If the `copy` parameter is set to True, the operation is 
        performed on a copy of the graph, leaving the original graph unchanged. 
        Otherwise, the operation is performed on the original graph.

        Parameters:
        -----------
        copy : bool, optional (default=True)
            If True, the operation is performed on a copy of the graph. If False, 
            the operation is performed on the original graph.

        Returns:
        --------
        networkx.Graph
            The graph with double edges removed.
        """
        if copy:
            G = self.G.copy()
        else:
            G = self.G
        for u, v, _ in G.edges(data=True):
            if G.has_edge(u, v) or G.has_edge(v, u):
                G.remove_edge(v, u)

        return G

    # Plotting functions
    def plot_map(self) -> folium.Map:
        """
        Plots a map using folium with nodes and edges from the graph data.
        This method creates a folium map and adds nodes and edges to it based on the 
        type of transport. If the transport type is not 'bus', it creates a feature 
        group for each line and adds the corresponding edges and nodes to the map. 
        If the transport type is 'bus', it uses folium GeoJSON to add the nodes and 
        edges to the map.
        Returns:
            folium.Map: A folium map with the plotted nodes and edges.
        """
        nodes, edges = self.get_gdf()

        # Create a folium map
        m = folium.Map(
            tiles="CartoDB positron", 
            location=[nodes['y'].mean(), nodes['x'].mean()], 
            zoom_start=12
        )

        default_color = 'gray'

        # If the kind of transport is not bus, create a feature group for each line
        if self.kind != 'bus':
            # Get the set of line names from the 'name' attribute of the edges
            # and convert it to a list
            # This will be used to create a feature group for each line
            lines = sorted(list(set(edges['ref'].dropna())))

            fg = folium.FeatureGroup(control=False)
            m.add_child(fg)

            # Add a child to the feature group for each line
            for line in lines:
                subgroup = folium.plugins.FeatureGroupSubGroup(fg, line.capitalize())
                m.add_child(subgroup)
                
                # Get the edges for the current line
                line_edges = edges[edges['ref'] == line]
                line_nodes = nodes[nodes['line'] == line]

                # Add the edges to the map
                for _, row in line_edges.iterrows():
                    line = folium.PolyLine(
                        locations=[(row['geometry'].coords[i][1], row['geometry'].coords[i][0]) for i in range(len(row['geometry'].coords))],
                        color=row.get('colour', default_color),
                        weight=5,
                        tooltip=folium.Tooltip(row['ref']),
                        popup=folium.Popup(row['ref'])
                    )
                    subgroup.add_child(line)

                # Add the nodes to the map
                for _, node_row in line_nodes.iterrows():
                    circle = folium.Circle(
                        location=[node_row['y'], node_row['x']],
                        radius=30,
                        color='black',
                        fill=True,
                        fill_color=node_row.get('colour', default_color),
                        tooltip=folium.Tooltip(node_row['name']),
                        popup=folium.Popup(node_row['name'])
                    )
                    subgroup.add_child(circle)

        # If the kind of transport is bus, use folium GeoJSON
        else:
            feature_collection_nodes = nodes[~(nodes.geometry.isna() | nodes.geometry.is_empty)].__geo_interface__
            feature_collection_edges = edges[~(edges.geometry.isna() | edges.geometry.is_empty)].__geo_interface__

            #folium.GeoJson(
            #    feature_collection_edges,
            #    name='Bus Lines',
            #    marker=folium.Circle(radius=30, fill_color="orange", fill_opacity=0.4, color="black", weight=1),
            #    tooltip=folium.GeoJsonTooltip(fields=['ref']),
            #    popup=folium.GeoJsonPopup(fields=['ref']),
            #    style_function=lambda x: {
            #        "fillColor": x.get('colour', default_color),
            #    },
            #    highlight_function=lambda x: {"fillOpacity": 0.8},
            #    zoom_on_click=True,
            #).add_to(m)

            folium.GeoJson(
                feature_collection_nodes,
                name='Bus Stops',
                marker=folium.Circle(radius=30, fill_color="orange", fill_opacity=0.4, color=default_color, weight=1),
                tooltip=folium.GeoJsonTooltip(fields=['name', 'line']),
                popup=folium.GeoJsonPopup(fields=['name', 'line']),
                style_function=lambda x: {
                    "fillColor": x.get('colour', default_color),
                    "radius": 30,
                },
                highlight_function=lambda x: {"fillOpacity": 0.8},
                zoom_on_click=True,
            ).add_to(m)

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

    def degree_distribution(self) -> None:
        """
        Plots and saves the degree distribution of the graph G.
        This method calculates the degree distribution of the graph G, fits a power law to the degree distribution,
        and plots the results in three subplots:
        1. Degree distribution with a power law fit.
        2. Normalized degree distribution.
        3. Normalized degree distribution in log-log scale.
        The plots are saved as an SVG file in the specified image directory.
        Returns:
            None
        """
        G = self.G
        degree_freq = nx.degree_histogram(G)
        degree_freq_norm = [x / sum(degree_freq) for x in degree_freq]
        degrees = range(len(degree_freq_norm))

        M = nx.to_scipy_sparse_matrix(G)
        xmin = min([d[1] for d in G.degree()])
        indegrees = M.sum(0).A[0]
        degree = np.bincount(indegrees)
        fit = powerlaw.Fit(np.array(degree)+1, fit_method='KS')
        alpha = fit.power_law.alpha
        
        plt.figure(figsize=(24, 8)) 

        plt.subplot(1, 3, 1)
        plt.plot(range(len(degree)),degree,'b.')   
        plt.loglog()
        plt.xlabel(r'Degree $d$')
        plt.ylabel(rf'P(k)')      
        plt.title(f'Degree Distribution (Power Law Fit, alpha={alpha:.2f})')    
        
        plt.subplot(1, 3, 2)
        plt.bar(degrees, degree_freq_norm, color='b', width=.5)
        plt.xlabel(r'Degree $d$')
        plt.ylabel(r'Fraction $p_d$ of nodes with degree $d$')
        plt.title('Degree Distribution')

        plt.subplot(1, 3, 3)
        plt.bar(degrees, degree_freq_norm, color='b', width=.5)
        plt.xlabel(r'Degree $d$')
        plt.ylabel(r'Fraction $p_d$ of nodes with degree $d$')
        plt.xscale('log')
        plt.yscale('log')
        plt.title('Degree Distribution (Log-Log)')

        plt.tight_layout()
        plt.savefig(self.img_dir / f"{self.location}_{self.kind}_degree_distribution_all.svg", format='svg', dpi=400)
        plt.show()

    def plot_dist(self) -> None:
        """
        Plots the Cumulative Distribution Function (CDF) and Complementary Cumulative Distribution Function (CCDF) 
        of the degree distribution of the graph G.
        The method calculates the degree sequence of the graph, fits a power-law distribution to the degree data, 
        and then plots the CDF and CCDF using matplotlib.
        The resulting plots are saved as an SVG file in the specified image directory.
        Parameters:
        None
        Returns:
        None
        """
        G = self.G
        degree_sequence = sorted([d for _, d in G.degree()], reverse=True)  # degree sequence
        degreeCount = Counter(degree_sequence)
        deg, cnt = zip(*degreeCount.items())
        fit = powerlaw.Fit(deg, xmin=1, fit_method='KS')

        plt.figure(figsize=(16, 8))
        
        plt.subplot(1, 2, 1)
        fit.plot_cdf(color='r',linestyle='--',label='fit cdf')
        plt.title("CDF plot")
        plt.ylabel("CDF")
        plt.xlabel("Degree x")
            
        plt.subplot(1, 2, 2)
        fit.power_law.plot_pdf(color='r',linestyle='--',label='fit ccdf')
        fit.plot_ccdf()
        plt.title("CCDF plot")
        plt.ylabel(r'Fraction $p_d$ of nodes with degree $d$ or greater')
        plt.xlabel(r'Degree $d$')
        
        plt.tight_layout()
        plt.savefig(self.img_dir / f"{self.location}_{self.kind}_cdf_ccdf.svg", format='svg', dpi=400)
        plt.show()



    def network_summary(self) -> dict:
        """
        Generate a summary of the network's key metrics and properties.
        Returns:
            dict: A dictionary containing the following network metrics:
                - max_degree (int): The highest degree of any node in the network.
                - hubs (dict): Hubs scores from HITS algorithm.
                - authorities (dict): Authorities scores from HITS algorithm.
                - pagerank (dict): PageRank scores of nodes.
                - density (float): The density of the network.
                - average_shortest_path (float): The average shortest path length in the network.
                - diameter (int): The diameter of the network.
                - avg_cluster (float): The average clustering coefficient of the network.
                - transitivity (float): The transitivity of the network.
                - bridges (int): The number of bridges in the network.
                - degree_centrality (dict): Degree centrality scores of nodes.
                - betweenness_centrality (dict): Betweenness centrality scores of nodes.
                - eigenvector_centrality (dict): Eigenvector centrality scores of nodes.
                - closeness_centrality (dict): Closeness centrality scores of nodes.
                - information_centrality (dict): Information centrality scores of nodes.
                - cliques (int): The number of cliques in the network.
                - connectedness (float): The connectedness value of the network.
                - fragmentation (float): The fragmentation value of the network.
        """
        G = self.G
        if not isinstance(G, nx.Graph):
            G = nx.Graph(G)

        # nodes with highest degree along with position
        degree_sequence = sorted(G.degree, key=lambda x: x[1], reverse=True)
        max_degree = degree_sequence[0][1]

        # average shortest path length
        if nx.is_connected(G):
            apl = nx.average_shortest_path_length(G, weight='length')
        else:
            net_cc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
            apl = nx.average_shortest_path_length(net_cc, weight='length')

        if nx.is_connected(G):
            diameter = nx.diameter(G)
        else:
            diameter = nx.diameter(net_cc)

        # eigenvecotr centrality
        correct = False
        iter = 5000
        while not correct:
            try:
                eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=iter, weight='length')
                correct = True
            except nx.PowerIterationFailedConvergence:
                print(f"Power iteration failed at {iter}, trying again.")
                iter += 5000

        # cliques
        cliques = sum(1 for c in nx.find_cliques(G))

        data = {
            "max_degree": max_degree,
            "density": nx.density(G),
            "average_shortest_path": apl,
            "diameter": diameter,
            "transitivity": nx.transitivity(G),
            "bridges": len(list(nx.bridges(G))),
            "degree_centrality": nx.degree_centrality(G),
            "betweenness_centrality": nx.betweenness_centrality(G, weight='length'),
            "eigenvector_centrality": eigenvector_centrality,
            "closeness_centrality": nx.closeness_centrality(G, distance='length'),
            "cliques": cliques
        }

        return data

    def basic_stats(self):
        G = self.G
        if isinstance(G, nx.Graph):
            G = nx.MultiDiGraph(G)
        return ox.basic_stats(G)

    def extended_stats(self):
        stats = dict()
        G = self.G
        avg_neighbor_degree = nx.average_neighbor_degree(G)
        stats["avg_neighbor_degree"] = avg_neighbor_degree
        stats["avg_neighbor_degree_avg"] = sum(avg_neighbor_degree.values()) / len(avg_neighbor_degree)
        avg_wtd_nbr_deg = nx.average_neighbor_degree(G, weight="length")
        stats["avg_weighted_neighbor_degree"] = avg_wtd_nbr_deg
        stats["avg_weighted_neighbor_degree_avg"] = sum(avg_wtd_nbr_deg.values()) / len(avg_wtd_nbr_deg)
        degree_centrality = nx.degree_centrality(G)
        stats["degree_centrality"] = degree_centrality
        stats["degree_centrality_avg"] = sum(degree_centrality.values()) / len(degree_centrality)
        stats["clustering_coefficient"] = nx.clustering(G)
        stats["clustering_coefficient_avg"] = nx.average_clustering(G)
        stats["clustering_coefficient_weighted"] = nx.clustering(G, weight="length")
        stats["clustering_coefficient_weighted_avg"] = nx.average_clustering(G, weight="length")
        stats["node_connectivity"] = nx.node_connectivity(G)
        stats["edge_connectivity"] = nx.edge_connectivity(G)
        stats["node_connectivity_avg"] = nx.average_node_connectivity(G)
        length_func = nx.single_source_dijkstra_path_length
        sp = {source: dict(length_func(G, source, weight="length")) for source in G.nodes}
        print("Calculated shortest path lengths")
        eccentricity = nx.eccentricity(G, sp=sp)
        stats["eccentricity"] = eccentricity
        diameter = nx.diameter(G, e=eccentricity)
        stats["diameter"] = diameter
        radius = nx.radius(G, e=eccentricity)
        stats["radius"] = radius
        center = nx.center(G, e=eccentricity)
        stats["center"] = center
        periphery = nx.periphery(G, e=eccentricity)
        stats["periphery"] = periphery
        return stats

    def compute_connectedness(self, G: nx.Graph, with_compactness: bool = False, verbose: bool = False) -> float:
        """
        Computes the connectedness of the graph G more efficiently.
        Args:
            G (nx.Graph): The input graph.
            with_compactness (bool): Whether to compute the compactness metric.
        Returns:
            float: The connectedness value (and compactness value if requested).
        """
        n = len(G.nodes)
        components = list(nx.connected_components(G))  # Get all connected components
        same_component_count = 0
        same_component_count_compactness = 0

        for component in tqdm(components, desc='Processing connected components', disable=not verbose):
            component_list = list(component)
            component_size = len(component_list)

            # Count pairs within this component
            same_component_count += component_size * (component_size - 1) / 2

            if with_compactness:
                # Compute compactness within this component
                for i, j in combinations(component_list, r=2):
                    if nx.has_path(G, source=i, target=j):
                        distance = ox.distance.great_circle(
                            G.nodes[i]['y'], G.nodes[i]['x'],
                            G.nodes[j]['y'], G.nodes[j]['x']
                        )
                        same_component_count_compactness += 1 / distance

        # Normalize connectedness
        total_pairs = n * (n - 1) / 2
        connectedness_value = same_component_count / total_pairs

        if with_compactness:
            compactness_value = same_component_count_compactness / total_pairs
            return connectedness_value, compactness_value

        return connectedness_value

    def small_world_summary(self) -> float:
        """
        Computes the small-world coefficient of the graph G.
        The small-world coefficient is calculated as the ratio of the average shortest path length
        of a random graph to the average shortest path length of the graph G, minus the ratio of
        the clustering coefficient of the graph G to the clustering coefficient of a regular graph.
        Returns:
            float: The small-world coefficient of the graph G.
        """
        G = self.G
        if not isinstance(G, nx.Graph):
            G = nx.Graph(G)
        
        n = len(G.nodes)
        m = len(G.edges)
        # get the average degree of the graph
        d = sum(dict(G.degree()).values()) / n
        er_graph = nx.erdos_renyi_graph(n=n, p=0.1, seed=self.SEED) # create a random graph with the same number of nodes and probability of 0.1
        regular_graph = nx.random_regular_graph(d=4, n=n, seed=self.SEED) # create a regular graph with the same number of nodes and degree of 4
        C = nx.average_clustering(G)
        if nx.is_connected(G):
            L = nx.average_shortest_path_length(G, weight='length')
        else:
            net_cc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
            L = nx.average_shortest_path_length(net_cc, weight='length')

        step1 = nx.average_shortest_path_length(er_graph) / L
        step2 = C / nx.average_clustering(regular_graph)
        return step1 - step2 # small world coefficient

    def random_node(self, G: nx.Graph) -> int:
        """
        Selects a random node from the graph.
        
        Parameters:
            G (nx.Graph): A NetworkX graph object.

        Returns:
            node: A randomly selected node from the graph.
        """
        return [np.random.choice(list(G.nodes()))]

    def dismantle(self, func, **args) -> Tuple[List[float], List[float]]:
        """
        Dismantles the graph by iteratively removing nodes based on a given function.

        Parameters:
        func (callable): A function that takes the graph and additional arguments, and returns a list of nodes to be removed.
        **args: Additional arguments to be passed to the function `func`.

        Returns:
        tuple: A tuple containing two lists:
            - removed_nodes (list of float): The fraction of nodes removed at each step.
            - components (list of float): The size of the largest connected component as a fraction of the total nodes at each step.
        """
        G = self.G.copy()
        total_nodes = G.number_of_nodes()
        removed_nodes = []
        connectedness_values = []
        while G.number_of_nodes() > total_nodes*0.6: # stop when 40% of nodes are removed
            node = func(G, **args)[0]
            G.remove_node(node)
            removed_nodes.append((len(removed_nodes)+1)/total_nodes)
            connectedness = self.compute_connectedness(G=G, with_compactness=False)
            connectedness_values.append(connectedness)
        return removed_nodes, connectedness_values

    def get_sorted_nodes(self, G: nx.Graph, score, weight: str = 'weight', reverse=True) -> List:
        """
        Sorts the nodes of the graph based on a given scoring function.

        Parameters:
        score (function): A function that takes a graph and returns a dictionary 
                          where keys are node identifiers and values are scores.
        reverse (bool, optional): If True, the nodes are sorted in descending order 
                                  of their scores. Defaults to True.

        Returns:
        list: A list of node identifiers sorted based on their scores.
        """
        nodes = score(G, weight=weight)
        if isinstance(nodes, dict):
            nodes = [(k, v) for k, v in nodes.items()]
        srt = sorted(nodes, key = lambda k: k[1], reverse=reverse)
        return [x[0] for x in srt]

    def get_top_n_nodes(self, summary, key, *args, **kwargs) -> List[Tuple]:
        """
        Get the top N nodes based on a specified key from the summary.

        Parameters:
        summary (dict): A dictionary containing node data.
        key (str): The key in the summary dictionary to sort by.
        n (int, optional): The number of top nodes to return. Default is 10.

        Returns:
        list: A list of tuples containing the top N nodes and their corresponding values.
        """
        n = kwargs.get('n')
        if n:
            return sorted(summary[key].items(), key=lambda x: x[1], reverse=True)[:n]
        return sorted(summary[key].items(), key=lambda x: x[1], reverse=True)


    def get_node_names(self, list_of_node_ids: List[Tuple], *args, **kwargs) -> List[str]:
        """
        Retrieve the names and additional information of nodes from a list of node IDs.

        Args:
            list_of_node_ids (List[Tuple]): A list of tuples where each tuple contains a node ID and an additional value.

        Returns:
            List[str]: A list of strings where each string contains the name of the node, the line it belongs to, 
                       and a Google Maps URL for its position.
        """
        nodes = []
        if n :=kwargs.get('n'):
            list_of_node_ids = list_of_node_ids[:n]
        for node_id, _ in list_of_node_ids:
            if '_' in str(node_id):
                node_id = node_id.split('_')[1]
            name = self.G.nodes[node_id].get('name')
            line = self.G.nodes[node_id].get('line')
            pos = self.G.nodes[node_id].get('pos')
            google_url = f"https://www.google.com/maps/search/?api=1&query={pos[1]},{pos[0]}"
            nodes.append(f"{name} ({line}) - {google_url}")
        return nodes
    
    def centrality_heatmap(self, centrality_measure: List[Tuple[int, float]]) -> folium.Map:
        """
        Plots the heatmap of the centrality measures using folium.
        Returns:
            None
        """
        # Get the nodes and edges as GeoDataFrames
        nodes, _ = self.get_gdf()

        # Normalize the centrality values
        vals = [val for node, val in centrality_measure]
        min_val = min(vals)
        max_val = max(vals)
        normalized_vals = {node: (val - min_val) / (max_val - min_val) for node, val in centrality_measure}

        # Create a folium map
        m = folium.Map(
            tiles="CartoDB positron", 
            location=[nodes['y'].mean(), nodes['x'].mean()], 
            zoom_start=12
        )

        cmap = cm.linear.OrRd_09.scale(0, 1)
        cmap.caption = 'Centrality Measure'
        m.add_child(cmap)
        gradient_dict = {}
        for ind_val, c in zip(cmap.index, cmap.colors):
            # Create gradient dictionary for heatmap on the fly
            r, g, b, a = c
            gradient_dict[ind_val] = f"rgba({r*255},{g*255},{b*255},{a})"

        # Add a heatmap layer
        folium.plugins.HeatMap(
            data=[[node[1]['y'], node[1]['x'], normalized_vals[node[1]['osmid']]] for node in nodes.iterrows()],
            min_opacity=0.1,
            radius=10,
            gradient=gradient_dict,
            blur=8,
        ).add_to(m)

        return m