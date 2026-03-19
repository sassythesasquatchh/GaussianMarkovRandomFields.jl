//! Mesh plotting helpers using `plotters`.
//!
//! These utilities provide lightweight SVG exports for FEM meshes and
//! node-wise fields so examples can mirror the Julia Makie recipes
//! without introducing heavy GUI dependencies.

use gmrf_core::Vector;
use gmrf_fem::mesh::{Mesh2d, Point2};
use plotters::prelude::*;

use crate::errors::VizError;

/// Visual styling for mesh plots.
#[derive(Debug, Clone)]
pub struct MeshPlotOptions {
    pub width: u32,
    pub height: u32,
    pub margin: u32,
    pub stroke_width: u32,
    pub show_nodes: bool,
    pub node_radius: u32,
}

impl Default for MeshPlotOptions {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            margin: 30,
            stroke_width: 2,
            show_nodes: true,
            node_radius: 4,
        }
    }
}

/// Additional styling for field plots.
#[derive(Debug, Clone)]
pub struct FieldPlotOptions {
    pub mesh: MeshPlotOptions,
    pub min_color: RGBColor,
    pub max_color: RGBColor,
}

impl Default for FieldPlotOptions {
    fn default() -> Self {
        Self {
            mesh: MeshPlotOptions::default(),
            min_color: BLUE,
            max_color: RED,
        }
    }
}

/// Render a mesh to an SVG, drawing element edges and optional node markers.
pub fn plot_mesh_svg(
    mesh: &Mesh2d,
    path: impl AsRef<std::path::Path>,
    options: MeshPlotOptions,
) -> Result<(), VizError> {
    let backend = SVGBackend::new(path.as_ref(), (options.width, options.height));
    let drawing_area = backend.into_drawing_area();
    drawing_area.fill(&WHITE)?;

    let bounds = Bounds::new(mesh.nodes());

    for element in mesh.elements() {
        let nodes = element.node_indices();
        for window in nodes.windows(2) {
            let start = bounds.project(&mesh.nodes()[window[0]], &options);
            let end = bounds.project(&mesh.nodes()[window[1]], &options);
            let style = ShapeStyle::from(&BLACK).stroke_width(options.stroke_width as u32);
            drawing_area.draw(&PathElement::new(vec![start, end], style))?;
        }
        if let (Some(first), Some(last)) = (nodes.first(), nodes.last()) {
            let start = bounds.project(&mesh.nodes()[*first], &options);
            let end = bounds.project(&mesh.nodes()[*last], &options);
            let style = ShapeStyle::from(&BLACK).stroke_width(options.stroke_width as u32);
            drawing_area.draw(&PathElement::new(vec![start, end], style))?;
        }
    }

    if options.show_nodes {
        for node in mesh.nodes() {
            let (x, y) = bounds.project(node, &options);
            drawing_area.draw(&Circle::new((x, y), options.node_radius, BLACK.filled()))?;
        }
    }

    drawing_area.present()?;
    Ok(())
}

/// Render a mesh and color nodes by a provided field, interpolating
/// between `min_color` and `max_color`.
pub fn plot_field_svg(
    mesh: &Mesh2d,
    field: &Vector,
    path: impl AsRef<std::path::Path>,
    options: FieldPlotOptions,
) -> Result<(), VizError> {
    if field.len() != mesh.num_nodes() {
        return Err(VizError::DimensionMismatch {
            expected: mesh.num_nodes(),
            actual: field.len(),
        });
    }

    let backend = SVGBackend::new(path.as_ref(), (options.mesh.width, options.mesh.height));
    let drawing_area = backend.into_drawing_area();
    drawing_area.fill(&WHITE)?;
    let bounds = Bounds::new(mesh.nodes());

    let (min_value, max_value) = field_min_max(field);

    for element in mesh.elements() {
        let nodes = element.node_indices();
        let polygon_points: Vec<_> = nodes
            .iter()
            .map(|&idx| bounds.project(&mesh.nodes()[idx], &options.mesh))
            .collect();

        let avg_value: f64 = nodes.iter().map(|&idx| field[idx]).sum::<f64>() / nodes.len() as f64;
        let fill = interpolate_color(
            avg_value,
            min_value,
            max_value,
            options.min_color,
            options.max_color,
        );

        drawing_area.draw(&Polygon::new(
            polygon_points.clone(),
            ShapeStyle {
                color: fill.to_rgba(),
                filled: true,
                stroke_width: options.mesh.stroke_width,
            },
        ))?;

        // Re-draw edges for clarity
        for window in polygon_points.windows(2) {
            let style = ShapeStyle::from(&BLACK).stroke_width(options.mesh.stroke_width);
            drawing_area.draw(&PathElement::new(window.to_vec(), style))?;
        }
        if let (Some(first), Some(last)) = (polygon_points.first(), polygon_points.last()) {
            let style = ShapeStyle::from(&BLACK).stroke_width(options.mesh.stroke_width);
            drawing_area.draw(&PathElement::new(vec![*last, *first], style))?;
        }
    }

    if options.mesh.show_nodes {
        for (idx, node) in mesh.nodes().iter().enumerate() {
            let (x, y) = bounds.project(node, &options.mesh);
            let fill = interpolate_color(
                field[idx],
                min_value,
                max_value,
                options.min_color,
                options.max_color,
            );
            drawing_area.draw(&Circle::new(
                (x, y),
                options.mesh.node_radius,
                ShapeStyle {
                    color: fill.to_rgba(),
                    filled: true,
                    stroke_width: 0,
                },
            ))?;
        }
    }

    drawing_area.present()?;
    Ok(())
}

#[derive(Debug)]
struct Bounds {
    min: Point2,
    max: Point2,
}

impl Bounds {
    fn new(nodes: &[Point2]) -> Self {
        let mut min = Point2::new(f64::INFINITY, f64::INFINITY);
        let mut max = Point2::new(f64::NEG_INFINITY, f64::NEG_INFINITY);

        for n in nodes {
            min.x = min.x.min(n.x);
            min.y = min.y.min(n.y);
            max.x = max.x.max(n.x);
            max.y = max.y.max(n.y);
        }

        Self { min, max }
    }

    fn project(&self, point: &Point2, options: &MeshPlotOptions) -> (i32, i32) {
        let width = (options.width - 2 * options.margin) as f64;
        let height = (options.height - 2 * options.margin) as f64;

        let span_x = (self.max.x - self.min.x).max(1e-9);
        let span_y = (self.max.y - self.min.y).max(1e-9);

        let norm_x = (point.x - self.min.x) / span_x;
        let norm_y = (point.y - self.min.y) / span_y;

        let x = options.margin as f64 + norm_x * width;
        let y = options.margin as f64 + (1.0 - norm_y) * height;

        (x.round() as i32, y.round() as i32)
    }
}

fn field_min_max(field: &Vector) -> (f64, f64) {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for v in field.iter() {
        min = min.min(*v);
        max = max.max(*v);
    }
    if min == f64::INFINITY {
        (0.0, 1.0)
    } else {
        (min, max)
    }
}

fn interpolate_color(value: f64, min: f64, max: f64, low: RGBColor, high: RGBColor) -> RGBColor {
    let span = (max - min).max(1e-12);
    let t = ((value - min) / span).clamp(0.0, 1.0);
    RGBColor(
        (low.0 as f64 + t * (high.0 as f64 - low.0 as f64)) as u8,
        (low.1 as f64 + t * (high.1 as f64 - low.1 as f64)) as u8,
        (low.2 as f64 + t * (high.2 as f64 - low.2 as f64)) as u8,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use gmrf_fem::mesh::ElementConnectivity;
    use tempfile::tempdir;

    fn demo_mesh() -> Mesh2d {
        let nodes = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];
        let elements = vec![
            ElementConnectivity::Triangle([0, 1, 2]),
            ElementConnectivity::Triangle([0, 2, 3]),
        ];
        Mesh2d::new(nodes, elements).unwrap()
    }

    #[test]
    fn writes_mesh_svg() {
        let mesh = demo_mesh();
        let dir = tempdir().unwrap();
        let path = dir.path().join("mesh.svg");

        plot_mesh_svg(&mesh, &path, MeshPlotOptions::default()).unwrap();
        let contents = std::fs::read_to_string(&path).unwrap();
        assert!(contents.contains("<svg"));
    }

    #[test]
    fn colors_field_by_value() {
        let mesh = demo_mesh();
        let dir = tempdir().unwrap();
        let path = dir.path().join("field.svg");

        let field = Vector::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
        plot_field_svg(&mesh, &field, &path, FieldPlotOptions::default()).unwrap();

        let contents = std::fs::read_to_string(&path).unwrap();
        assert!(contents.contains("polygon"));
    }

    #[test]
    fn rejects_mismatched_field_length() {
        let mesh = demo_mesh();
        let field = Vector::from_element(2, 0.0);
        let dir = tempdir().unwrap();
        let path = dir.path().join("bad.svg");
        let err = plot_field_svg(&mesh, &field, &path, FieldPlotOptions::default()).unwrap_err();
        matches!(err, VizError::DimensionMismatch { .. });
    }
}
