use nalgebra::{DMatrix, DVector};

/// Weighted undirected edge between two nodes.
#[derive(Debug, Clone, PartialEq)]
pub struct Edge {
    pub from: usize,
    pub to: usize,
    pub weight: f64,
}

/// Sparse adjacency description that can be transformed into a Laplacian
/// precision matrix.
#[derive(Debug, Clone, Default)]
pub struct WeightedGraph {
    nodes: usize,
    edges: Vec<Edge>,
}

impl WeightedGraph {
    /// Create a new empty graph with `nodes` vertices.
    pub fn new(nodes: usize) -> Self {
        Self {
            nodes,
            edges: Vec::new(),
        }
    }

    /// Add an undirected edge with the provided weight.
    pub fn add_edge(&mut self, from: usize, to: usize, weight: f64) {
        self.edges.push(Edge { from, to, weight });
    }

    /// Construct the combinatorial Laplacian associated with the graph.
    /// This matrix is symmetric and can be used as a precision matrix
    /// for intrinsic GMRF models.
    pub fn laplacian(&self, jitter: f64) -> DMatrix<f64> {
        let mut matrix = DMatrix::zeros(self.nodes, self.nodes);

        for edge in &self.edges {
            let Edge { from, to, weight } = *edge;
            matrix[(from, from)] += weight;
            matrix[(to, to)] += weight;
            matrix[(from, to)] -= weight;
            matrix[(to, from)] -= weight;
        }

        if jitter > 0.0 {
            for i in 0..self.nodes {
                matrix[(i, i)] += jitter;
            }
        }

        matrix
    }

    /// Convenience function to build a graph from an adjacency matrix.
    /// Entries on the diagonal are ignored.
    pub fn from_adjacency(adjacency: &DMatrix<f64>) -> Self {
        let nodes = adjacency.nrows();
        let mut graph = Self::new(nodes);
        for i in 0..nodes {
            for j in (i + 1)..nodes {
                let weight = adjacency[(i, j)];
                if weight != 0.0 {
                    graph.add_edge(i, j, weight);
                }
            }
        }
        graph
    }

    /// Create a mean-zero Gaussian Markov random field using the Laplacian
    /// of this graph as precision.
    pub fn as_gmrf(&self, jitter: f64) -> (DVector<f64>, DMatrix<f64>) {
        (DVector::zeros(self.nodes), self.laplacian(jitter))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn builds_laplacian() {
        let mut graph = WeightedGraph::new(3);
        graph.add_edge(0, 1, 2.0);
        graph.add_edge(1, 2, 1.0);

        let lap = graph.laplacian(0.1);
        assert_relative_eq!(lap[(0, 0)], 2.1);
        assert_relative_eq!(lap[(1, 1)], 3.1);
        assert_relative_eq!(lap[(2, 2)], 1.1);
        assert_relative_eq!(lap[(0, 1)], -2.0);
        assert_relative_eq!(lap[(1, 2)], -1.0);
    }
}
