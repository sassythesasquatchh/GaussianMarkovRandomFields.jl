//! Minimal Gmsh importer for 2D meshes.
//!
//! This parser supports ASCII `.msh` v2-style files with triangle (type 2) and quadrilateral
//! (type 3) elements. It is intentionally conservative but mirrors the Ferrite helpers used in the
//! Julia package for quick prototyping.

use std::fs;
use std::path::Path;

use crate::errors::FemError;
use crate::mesh::{ElementConnectivity, Mesh2d, Point2};

/// Parsed mesh from a Gmsh file.
#[derive(Debug, Clone)]
pub struct GmshMesh {
    pub mesh: Mesh2d,
}

impl GmshMesh {
    /// Load a mesh from the provided path.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, FemError> {
        let contents = fs::read_to_string(path)?;
        Self::from_str(&contents)
    }

    /// Parse a mesh from raw string contents.
    pub fn from_str(contents: &str) -> Result<Self, FemError> {
        let mut lines = contents.lines();

        // Parse nodes
        let mut nodes: Vec<Point2> = Vec::new();
        while let Some(line) = lines.next() {
            if line.trim() == "$Nodes" {
                let count: usize = lines
                    .next()
                    .ok_or_else(|| FemError::GmshParse("missing node count".into()))?
                    .trim()
                    .parse()
                    .map_err(|_| FemError::GmshParse("invalid node count".into()))?;
                for _ in 0..count {
                    let node_line = lines
                        .next()
                        .ok_or_else(|| FemError::GmshParse("unexpected end of nodes".into()))?;
                    let parts: Vec<&str> = node_line.split_whitespace().collect();
                    if parts.len() < 4 {
                        return Err(FemError::GmshParse("invalid node record".into()));
                    }
                    let x: f64 = parts[1]
                        .parse()
                        .map_err(|_| FemError::GmshParse("invalid node coordinate".into()))?;
                    let y: f64 = parts[2]
                        .parse()
                        .map_err(|_| FemError::GmshParse("invalid node coordinate".into()))?;
                    nodes.push(Point2::new(x, y));
                }

                // consume $EndNodes
                lines
                    .next()
                    .filter(|l| l.trim() == "$EndNodes")
                    .ok_or_else(|| FemError::GmshParse("missing $EndNodes".into()))?;
                break;
            }
        }

        // Parse elements
        let mut elements: Vec<ElementConnectivity> = Vec::new();
        while let Some(line) = lines.next() {
            if line.trim() == "$Elements" {
                let count: usize = lines
                    .next()
                    .ok_or_else(|| FemError::GmshParse("missing element count".into()))?
                    .trim()
                    .parse()
                    .map_err(|_| FemError::GmshParse("invalid element count".into()))?;
                for _ in 0..count {
                    let elem_line = lines
                        .next()
                        .ok_or_else(|| FemError::GmshParse("unexpected end of elements".into()))?;
                    let parts: Vec<&str> = elem_line.split_whitespace().collect();
                    if parts.len() < 5 {
                        continue;
                    }
                    let elem_type: i32 = parts[1]
                        .parse()
                        .map_err(|_| FemError::GmshParse("invalid element type".into()))?;
                    let num_tags: usize = parts[2]
                        .parse()
                        .map_err(|_| FemError::GmshParse("invalid tag count".into()))?;
                    let node_start = 3 + num_tags;
                    if elem_type == 2 {
                        if parts.len() < node_start + 3 {
                            return Err(FemError::GmshParse("triangle missing nodes".into()));
                        }
                        let nodes_idx = [
                            parts[node_start],
                            parts[node_start + 1],
                            parts[node_start + 2],
                        ]
                        .map(|p| {
                            p.parse::<usize>()
                                .map_err(|_| FemError::GmshParse("invalid node index".into()))
                        });
                        let parsed: Vec<usize> =
                            nodes_idx.into_iter().collect::<Result<Vec<_>, _>>()?;
                        elements.push(ElementConnectivity::Triangle([
                            parsed[0] - 1,
                            parsed[1] - 1,
                            parsed[2] - 1,
                        ]));
                    } else if elem_type == 3 {
                        if parts.len() < node_start + 4 {
                            return Err(FemError::GmshParse("quadrilateral missing nodes".into()));
                        }
                        let nodes_idx = [
                            parts[node_start],
                            parts[node_start + 1],
                            parts[node_start + 2],
                            parts[node_start + 3],
                        ]
                        .map(|p| {
                            p.parse::<usize>()
                                .map_err(|_| FemError::GmshParse("invalid node index".into()))
                        });
                        let parsed: Vec<usize> =
                            nodes_idx.into_iter().collect::<Result<Vec<_>, _>>()?;
                        elements.push(ElementConnectivity::Quadrilateral([
                            parsed[0] - 1,
                            parsed[1] - 1,
                            parsed[2] - 1,
                            parsed[3] - 1,
                        ]));
                    }
                }

                break;
            }
        }

        if nodes.is_empty() || elements.is_empty() {
            return Err(FemError::GmshParse("no nodes or elements parsed".into()));
        }

        let mesh = Mesh2d::new(nodes, elements)?;
        Ok(Self { mesh })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn parses_simple_triangle_mesh() {
        let msh = r#"$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
3
1 0 0 0
2 1 0 0
3 0 1 0
$EndNodes
$Elements
1
1 2 0 1 2 3
$EndElements
"#;

        let gmsh = GmshMesh::from_str(msh).unwrap();
        assert_eq!(gmsh.mesh.num_nodes(), 3);
        assert_eq!(gmsh.mesh.num_elements(), 1);
    }

    #[test]
    fn parses_from_file() {
        let msh = r#"$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
4
1 0 0 0
2 1 0 0
3 1 1 0
4 0 1 0
$EndNodes
$Elements
1
1 3 0 1 2 3 4
$EndElements
"#;
        let mut tmp = NamedTempFile::new().unwrap();
        use std::io::Write;
        tmp.write_all(msh.as_bytes()).unwrap();

        let gmsh = GmshMesh::from_file(tmp.path()).unwrap();
        assert_eq!(gmsh.mesh.num_nodes(), 4);
        assert_eq!(gmsh.mesh.num_elements(), 1);
    }
}
