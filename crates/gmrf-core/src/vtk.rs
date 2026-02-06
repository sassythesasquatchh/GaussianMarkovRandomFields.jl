//! VTK writers for visualizing GMRF fields on regular grids.
//!
//! This mirrors the Julia tutorials where posterior samples are exported for
//! external visualization (e.g., ParaView) while keeping a sparse-first workflow.

use crate::types::Vector;
use std::io::{self, Write};

/// Write scalar fields on a structured grid as legacy VTK STRUCTURED_POINTS (ASCII).
pub fn write_structured_points<W: Write>(
    writer: &mut W,
    grid_size: usize,
    fields: &[(&str, &Vector)],
) -> io::Result<()> {
    if grid_size == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "grid_size must be positive",
        ));
    }
    let npoints = grid_size * grid_size;
    let spacing = if grid_size > 1 {
        1.0 / (grid_size as f64 - 1.0)
    } else {
        1.0
    };

    writeln!(writer, "# vtk DataFile Version 3.0")?;
    writeln!(writer, "GMRF structured grid")?;
    writeln!(writer, "ASCII")?;
    writeln!(writer, "DATASET STRUCTURED_POINTS")?;
    writeln!(writer, "DIMENSIONS {} {} {}", grid_size, grid_size, 1)?;
    writeln!(writer, "ORIGIN 0 0 0")?;
    writeln!(writer, "SPACING {:.6} {:.6} 1", spacing, spacing)?;
    writeln!(writer, "POINT_DATA {}", npoints)?;

    for (name, values) in fields {
        if values.len() != npoints {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "field '{}' has length {}, expected {}",
                    name,
                    values.len(),
                    npoints
                ),
            ));
        }
        writeln!(writer, "SCALARS {} float 1", name)?;
        writeln!(writer, "LOOKUP_TABLE default")?;
        for y in 0..grid_size {
            for x in 0..grid_size {
                let idx = y * grid_size + x;
                writeln!(writer, "{:.6}", values[idx])?;
            }
        }
    }

    Ok(())
}
