//! Neighborhood graph helpers for spatial latent models.

use petgraph::graph::{NodeIndex, UnGraph};

use crate::errors::LatentModelError;

/// Undirected neighbor graph for conditional autoregressive priors.
pub struct Neighborhood {
    graph: UnGraph<(), ()>,
    nodes: Vec<NodeIndex>,
}

impl Neighborhood {
    /// Build a graph with a fixed number of nodes and undirected edges.
    pub fn from_edges(
        node_count: usize,
        edges: &[(usize, usize)],
    ) -> Result<Self, LatentModelError> {
        if node_count == 0 {
            return Err(LatentModelError::InvalidGraph(
                "neighborhood must contain at least one node",
            ));
        }

        for &(u, v) in edges {
            if u >= node_count || v >= node_count {
                return Err(LatentModelError::InvalidGraph(
                    "edge indices must be within node count",
                ));
            }
        }

        let mut graph = UnGraph::<(), ()>::new_undirected();
        let nodes: Vec<_> = (0..node_count).map(|_| graph.add_node(())).collect();
        for &(u, v) in edges {
            graph.add_edge(nodes[u], nodes[v], ());
        }

        Ok(Self { graph, nodes })
    }

    /// Number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Degree for each node in insertion order.
    pub fn degrees(&self) -> Vec<usize> {
        self.nodes
            .iter()
            .map(|&n| self.graph.neighbors(n).count())
            .collect()
    }

    /// Undirected edge list as zero-based index pairs.
    pub fn edges(&self) -> Vec<(usize, usize)> {
        self.graph
            .edge_indices()
            .filter_map(|e| self.graph.edge_endpoints(e))
            .map(|(a, b)| (a.index(), b.index()))
            .collect()
    }
}
