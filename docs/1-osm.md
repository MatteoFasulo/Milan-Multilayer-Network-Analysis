# OpenStreetMap

Using data from [OpenStreetMap (OSM)](http://www.openstreetmap.org/), we construct the metro, tram and bus networks for Milan (Italy). We downloaded data in geo-referenced vectorial format from Open Street Map under the Open Database License (ODbL) v1.0. Specifically, the datasets were extracted using the [Overpass Turbo](https://overpass-turbo.eu/) tool by invoking its API. All data used in this study is publicly available, ensuring transparency and accessibility.

## Query

The following query in Overpass QL format was used to extract the data for Milan public transport system:

``` py title="osmgraph.py"
[out:json]
[timeout:30];
area["name:en"="Milan"]->.searchArea;
relation["type"="route"]["route"="bus"]["operator"="ATM"](area.searchArea);
relation["type"="route"]["route"="tram"](area.searchArea);
relation["type"="route"]["route"="subway"](area.searchArea);
out meta;
>;
out body;
```

## Data Processing

We have extracted the route elements from the OSM data using the following tags:

``` py title="osmgraph.py"
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
```

And for the nodes:

``` py title="osmgraph.py"
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
```

the main algorithm for creating the graph is as follows:

``` py title="osmgraph.py"
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
```

### Simplification of the Graph

Since the raw data extracted from OSM can be quite detailed and contain redundant information, we provide a method to simplify the graph by clustering nodes within a certain distance threshold and creating new edges between the cluster centers. This process reduces the complexity of the graph while preserving its essential structure. The nodes within each cluster are merged into a single representative node, and the edges are updated accordingly.

In order to find the clusters of nodes that are close to each other, we use the DBSCAN clustering algorithm, which groups together nodes that are within a specified distance threshold. The distance threshold is given in meters and determines the maximum distance between two nodes for them to be considered part of the same cluster. Then a constant factor is used to convert the distance threshold from meters to a suitable range for the clustering algorithm parameter (epsilon).

The simplification algorithm is as follows:

``` py title="osmgraph.py"
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
```