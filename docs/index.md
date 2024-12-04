# Introduction


This repository contains the code and report for the "Social Network Analysis" course at Alma Mater Studiorum University of Bologna.

## Authors
- [Matteo Fasulo](https://github.com/MatteoFasulo)
- [Luca Babboni](https://github.com/ElektroDuck)
- [Maksim Omelchenko](https://github.com/omemaxim)
- [Francesca Bertoglio](https://github.com/francescabertoglio)

## Report

The PDF report is available [here](https://matteofasulo.github.io/Milan-Multilayer-Network-Analysis/report.pdf).

## Abstract 
In recent decades, analyzing the robustness of Public Transport Networks (PTNs) has become increasingly critical for ensuring their reliability and efficiency. As urban populations grow and transportation systems become more interconnected, understanding the resilience of PTNs to disruptions is essential for effective planning, optimization, and sustainable development. Complex network theory provides a powerful framework for investigating the structural and functional properties of spatial networks like PTNs, offering insights into their vulnerability and capacity to withstand failures.

Our analysis examines the multilayer PTN of Milan, Italy, focusing on three key aspects: 
1. the spatial distribution of centrality measures, specifically betweenness centrality (BC) and closeness centrality (CC)
2. the small-worldness properties of the network
3. its robustness through what-if scenarios simulating node removal.

These features help identify critical nodes that underpin the network's resilience.
The network comprises three interconnected layers: metro, tram, and bus systems. Each layer is represented as a graph in L-space topology [1], where stops and stations are nodes, and their connections are edges (e.g., a bus traveling between two stops). Data for this study were sourced from Open Street Map, and the full analysis, including a Python notebook, is publicly available in this repo

## Dataset

Using data from [OpenStreetMap (OSM)](http://www.openstreetmap.org/), we construct the metro, tram and bus networks for Milan (Italy). We downloaded data in geo-referenced vectorial format from Open Street Map under the Open Database License (ODbL) v1.0. Specifically, the datasets were extracted using the [Overpass Turbo](https://overpass-turbo.eu/) tool by invoking its API. All data used in this study is publicly available, ensuring transparency and accessibility.

For storage and manipulation, the data was structured as a graph using the [NetworkX](https://networkx.org/) library, a Python package designed for creating, analyzing, and visualizing complex networks. NetworkX offers a rich suite of tools for computing various graph measures and performing network analysis. Additionally, other Python libraries, such as [OSMnx](https://osmnx.readthedocs.io/en/stable/) and [GeoPandas](https://geopandas.org/en/stable/), were employed to streamline the processing pipeline and enhance code readability while maintaining high development efficiency. In addition, a series of automatic and manual topological cleaning operations were neeeded in order to extract consistent and usable graphs.

Three distinct datasets were collected for this study, representing different modes of public transport in Milan: metro, tram, and bus (including both stops and routes). Of these, the bus dataset is the most complex due to the extensive network. For consistency and relevance, only bus stops operated by Azienda Trasporti Milanesi S.p.A (ATM) were included. 

The datasets were further categorized as follows:

| Layer |   N  |   M  | Lines |
|:-----:|:----:|:----:|:-----:|
| Metro |  125 |  258 |   5   |
|  Tram |  308 |  680 |   17  |
|  Bus  | 1370 | 3788 |  113  |


## References
- [1] Sen et al, "Small-world properties of the Indian railway network", Mar/2023, American Physical Society, doi=10.1103/PhysRevE.67.036106, url=https://link.aps.org/doi/10.1103/PhysRevE.67.036106

## License

This project is licensed under the [MIT License](https://matteofasulo.github.io/Milan-Multilayer-Network-Analysis/LICENSE).
