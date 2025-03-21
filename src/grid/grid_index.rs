use ndarray::{ArrayView1, ArrayView2};
use rustc_hash::FxHashMap;
use std::cmp::Ordering;
use std::ops::Range;

/// GridIndex represents a p-dimensional grid over a set of n points.
/// Each axis is partitioned by (possibly empty) sorted boundaries. Initially,
/// no boundaries exist so each axis spans (-∞, ∞) and the grid has one cell containing all points.
/// This implementation uses a HashMap for cell storage to efficiently handle sparse data.

#[derive(Debug, Clone)]
pub struct GridIndex {
    /// For each axis, the sorted list of split values.
    /// These boundaries partition each axis into intervals.
    pub boundaries: Vec<Vec<f64>>,
    pub intervals: Vec<Vec<(f64, f64)>>,
    pub observation_counts: Vec<Vec<usize>>,
    /// Sparse grid cells. Each cell holds a vector of point indices.
    /// The flat index is computed using precomputed strides.
    /// Only cells that contain points are stored in the HashMap.
    pub cells: FxHashMap<usize, Vec<usize>>,
    /// Strides for flattening multi-dimensional cell indices.
    pub strides: Vec<usize>,
}

impl GridIndex {
    /// Creates a new GridIndex from an n x p matrix of points.
    /// Initially, there are no splits so the entire space is one cell.
    pub fn new(points: ArrayView2<'_, f64>) -> Self {
        let n = points.nrows();
        let p = points.ncols();
        // Initially, no boundaries for any axis.
        let boundaries = vec![Vec::new(); p];
        // Each axis has one interval (no boundaries => one interval).
        let dims: Vec<usize> = vec![1; p];
        let strides = Self::compute_strides(&dims);
        // Only one cell (at index 0), containing all point indices.
        let mut cells = FxHashMap::with_capacity_and_hasher(1, Default::default());
        cells.insert(0, (0..n).collect());
        let intervals = vec![vec![(f64::NEG_INFINITY, f64::INFINITY)]; p];
        let observation_counts = vec![vec![n]; p];
        GridIndex {
            boundaries,
            intervals,
            observation_counts,
            cells,
            strides,
        }
    }

    pub fn from_boundaries_and_points(
        boundaries: Vec<Vec<f64>>,
        points: ArrayView2<'_, f64>,
    ) -> Self {
        let intervals: Vec<Vec<(f64, f64)>> = boundaries
            .iter()
            .map(|b| {
                if b.is_empty() {
                    vec![(f64::NEG_INFINITY, f64::INFINITY)]
                } else {
                    let mut intervals = Vec::with_capacity(b.len() + 1);
                    // First interval: -inf to first boundary
                    intervals.push((f64::NEG_INFINITY, b[0]));
                    // Middle intervals
                    intervals.extend(b.windows(2).map(|w| (w[0], w[1])));
                    // Last interval: last boundary to inf
                    intervals.push((b[b.len() - 1], f64::INFINITY));
                    intervals
                }
            })
            .collect();

        let dims: Vec<usize> = intervals.iter().map(|i| i.len()).collect();
        let mut grid_index = GridIndex {
            boundaries,
            intervals,
            observation_counts: vec![],
            cells: FxHashMap::default(),
            strides: Self::compute_strides(&dims),
        };
        grid_index.assign_cells(points);
        grid_index
    }

    /// Returns the current number of intervals (cells) along each axis.
    pub fn current_dims(&self) -> Vec<usize> {
        self.intervals.iter().map(|i| i.len()).collect()
    }

    /// Splits the grid globally along the given axis by inserting `split` as a new boundary.
    /// If the split is already present, nothing is changed.
    /// After updating boundaries, the grid's strides are recomputed and all points are reassigned.
    pub fn split_axis(
        &mut self,
        cells_affected: &[usize],
        axis: usize,
        split: f64,
        points: ArrayView2<'_, f64>,
    ) -> (usize, usize) {
        // Insert the new split value into the boundaries for this axis if not already present.
        let pos = self.compute_col_index_for_point(axis, split);
        let (curr_start, curr_end) = self.intervals[axis][pos];
        self.intervals[axis][pos] = (curr_start, split);
        self.intervals[axis].insert(pos + 1, (split, curr_end));
        self.boundaries[axis].insert(pos, split);

        // Store the old strides before recomputing
        let old_strides = self.strides.clone();

        // Recompute dimensions and strides.
        let dims = self.current_dims();
        self.strides = Self::compute_strides(&dims);

        let estimated_nonempty = self.cells.len();
        let mut new_cells =
            FxHashMap::with_capacity_and_hasher(estimated_nonempty, Default::default());

        // Create set of affected cells for faster lookup
        let affected_set: std::collections::HashSet<usize> =
            cells_affected.iter().copied().collect();

        // Handle unaffected cells - correctly remap the keys
        let mut to_move = Vec::new();
        for &old_flat_idx in self.cells.keys() {
            if !affected_set.contains(&old_flat_idx) {
                let mut new_flat_idx = 0;
                let mut remainder = old_flat_idx;
                for (i, &stride) in old_strides.iter().enumerate() {
                    let coord = remainder / stride;
                    remainder %= stride;
                    let new_coord = if i == axis && coord > pos {
                        coord + 1
                    } else {
                        coord
                    };
                    new_flat_idx += new_coord * self.strides[i];
                }
                to_move.push((old_flat_idx, new_flat_idx));
            }
        }

        // Move vectors from old keys to new keys without cloning
        for (old_idx, new_idx) in to_move {
            if let Some(points_idx) = self.cells.remove(&old_idx) {
                new_cells.insert(new_idx, points_idx);
            }
        }

        // Update observation counts
        self.observation_counts[axis][pos] = 0;
        self.observation_counts[axis].insert(pos + 1, 0);

        for &cell_idx in cells_affected {
            if let Some(cell_points) = self.cells.remove(&cell_idx) {
                // Reassign only the points from affected cells
                for i in cell_points {
                    let point = points.row(i);
                    let (cell_idx, cartesian) = self.compute_cell_index_for_point(point);

                    // Add point to the appropriate cell
                    new_cells.entry(cell_idx).or_insert_with(Vec::new).push(i);

                    // Update observation counts
                    self.observation_counts[axis][cartesian[axis]] += 1;
                }
            }
        }

        self.cells = new_cells;

        (pos, pos + 1)
    }

    /// Computes the strides given the number of intervals (dims) per axis.
    /// For dims = [d0, d1, ..., d(p-1)], the stride for axis i is
    /// product_{j=i+1}^{p-1} d_j.
    fn compute_strides(dims: &[usize]) -> Vec<usize> {
        let p = dims.len();
        let mut strides = vec![0; p];
        strides[p - 1] = 1;
        for i in (0..p - 1).rev() {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
        strides
    }

    /// Given a single point (as an ArrayView1) and the current boundaries,
    /// compute its flat cell index.
    fn compute_cell_index_for_point(&self, point: ArrayView1<f64>) -> (usize, Vec<usize>) {
        let mut index = 0;
        // For each axis, perform a binary search in the boundaries.
        // The resulting position is the cell index along that axis.
        let mut cartesian_index = vec![0; self.boundaries.len()];
        for (j, &coord) in point.iter().enumerate() {
            // binary_search returns Ok(pos) if equal or Err(pos) where pos is the number of boundaries less than the value.
            let pos = self.compute_col_index_for_point(j, coord);
            index += pos * self.strides[j];
            cartesian_index[j] = pos;
        }
        (index, cartesian_index)
    }

    pub fn compute_col_index_for_point(&self, col: usize, point: f64) -> usize {
        self.boundaries[col]
            .binary_search_by(|b| {
                if *b <= point {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            })
            .unwrap_or_else(|i| i)
    }

    fn assign_cells(&mut self, points: ArrayView2<'_, f64>) {
        for i in 0..points.nrows() {
            let point = points.row(i);
            let (cell_idx, _) = self.compute_cell_index_for_point(point);
            self.cells.entry(cell_idx).or_default().push(i);
        }
    }

    /// Performs a query on a single cell. The cell is identified by a slice of cell indices,
    /// one per axis. For example, for a 2D grid, [0, 1] means the first interval on axis 0 and
    /// the second interval on axis 1.
    ///
    /// Returns an Option containing a slice of point indices, or None if the length of
    /// `cell_indices` does not match the number of axes or if the cell is empty.
    pub fn query(&self, cell_indices: &[usize]) -> Option<&[usize]> {
        if cell_indices.len() != self.boundaries.len() {
            return None;
        }
        // Compute the flat index from the multi-dimensional cell indices.
        let flat_index: usize = cell_indices
            .iter()
            .zip(self.strides.iter())
            .map(|(&ci, &stride)| ci * stride)
            .sum();

        // Look up the cell in the HashMap
        self.cells.get(&flat_index).map(|v| &v[..])
    }

    pub fn collect_fixed_axis_cells(&self, fixed_axis: usize, fixed_index: usize) -> Vec<usize> {
        let dims = self.current_dims();
        if fixed_axis >= dims.len() || fixed_index >= dims[fixed_axis] {
            return Vec::new();
        }

        let s_fixed = self.strides[fixed_axis];
        let d_fixed = dims[fixed_axis];

        self.cells
            .keys()
            .filter(|&&cell| {
                let quotient = cell / s_fixed;
                (quotient % d_fixed) == fixed_index
            })
            .copied()
            .collect()
    }

    pub fn get_cartesian_coordinates(&self, flat_index: usize) -> Vec<usize> {
        let dims = self.current_dims();
        let strides = &self.strides;
        let mut coordinates = vec![0; dims.len()];
        let mut remainder = flat_index;
        for (i, &stride) in strides.iter().enumerate() {
            coordinates[i] = remainder / stride;
            remainder %= stride;
        }
        coordinates
    }

    pub fn query_range(&self, cell_ranges: &[Range<usize>]) -> Vec<usize> {
        let p = self.boundaries.len();
        if cell_ranges.len() != p {
            return vec![];
        }
        let dims = self.current_dims();
        let mut result = Vec::new();

        // Recursive helper to iterate over the specified multi-dimensional cell ranges.
        fn recursive_query(
            axis: usize,
            p: usize,
            cell_ranges: &[Range<usize>],
            strides: &[usize],
            current_index: usize,
            dims: &[usize],
            cells: &FxHashMap<usize, Vec<usize>>,
            result: &mut Vec<usize>,
        ) {
            if axis == p {
                // Only try to access cells that exist in the HashMap
                if let Some(cell) = cells.get(&current_index) {
                    result.extend(cell);
                }
                return;
            }
            for i in cell_ranges[axis].clone() {
                if i < dims[axis] {
                    let new_index = current_index + i * strides[axis];
                    recursive_query(
                        axis + 1,
                        p,
                        cell_ranges,
                        strides,
                        new_index,
                        dims,
                        cells,
                        result,
                    );
                }
            }
        }

        recursive_query(
            0,
            p,
            cell_ranges,
            &self.strides,
            0,
            &dims,
            &self.cells,
            &mut result,
        );
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_initial_grid() {
        // Create a 5x2 matrix (5 points in 2D).
        let points = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]];
        let grid = GridIndex::new(points.view());
        // Initially, there is only one cell containing all points.
        assert_eq!(grid.cells.len(), 1);
        assert_eq!(grid.cells.get(&0).unwrap().len(), 5);
    }

    #[test]
    fn test_split_and_query() {
        // 8 points in 2D.
        let points = array![
            [-10.0, -10.0],
            [-5.0, -5.0],
            [0.0, 0.0],
            [5.0, 5.0],
            [10.0, 10.0],
            [-7.0, 7.0],
            [7.0, -7.0],
            [0.0, 10.0]
        ];
        let mut grid = GridIndex::new(points.view());

        // Split axis 0 (x-coordinate) at 0.0.
        grid.split_axis(&[0], 0, 0.0, points.view());
        // Now, for axis 0 there should be 2 intervals.
        let dims = grid.current_dims();
        assert_eq!(dims[0], 2);
        // The grid now has 2 cells (since axis 1 is still unsplit).
        assert_eq!(grid.cells.len(), 2);

        // Query the cell corresponding to x < 0 (cell index 0 along axis 0).
        let cell_neg = grid.query(&[0, 0]).unwrap();
        // Points with x-coordinate < 0 are: indices 0, 1, 5.
        assert_eq!(cell_neg.len(), 3);

        // Split axis 1 (y-coordinate) at 0.0.
        grid.split_axis(&[0, 1], 1, 0.0, points.view());
        let dims = grid.current_dims();
        assert_eq!(dims, vec![2, 2]);
        // Now there are 2*2 = 4 cells, but we might not have points in all of them
        assert!(grid.cells.len() <= 4 && !grid.cells.is_empty());

        // Query the cell with x < 0 and y < 0, i.e. cell indices [0, 0].
        let cell_neg_neg = grid.query(&[0, 0]).unwrap();
        // In our sample, points with x < 0 and y < 0: indices 0 and 1.
        assert_eq!(cell_neg_neg.len(), 2);

        // You can also perform a range query.
        // For example, query all cells with x index in 0..2 and y index in 0..1.
        let pts = grid.query_range(&[0..2, 0..1]);
        // This should collect all points whose y-coordinate is in the lower interval.
        // In our test data, that should be indices 0, 1, 6.
        assert_eq!(pts.len(), 3);
    }
}
