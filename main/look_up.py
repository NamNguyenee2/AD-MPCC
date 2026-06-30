import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline, LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import pickle


data = pd.read_csv("waypoints.csv")

# Sort by theta and merge duplicates
df = data[["theta", "X", "Y"]].copy()
df = df.sort_values("theta")
df = df.groupby("theta", as_index=False).mean()

theta = df["theta"].to_numpy()
x = df["X"].to_numpy()
y = df["Y"].to_numpy()

print(x[0])
print(y[0])


dx = np.diff(x)
dy = np.diff(y)

segment_lengths = np.sqrt(dx**2 + dy**2)

s = np.concatenate([[0], np.cumsum(segment_lengths)])
spline_x = CubicSpline(s, x, bc_type='periodic')
spline_y = CubicSpline(s, y, bc_type='periodic')

# Create cubic splines
# spline_x = CubicSpline(theta, x, bc_type='natural')
# spline_y = CubicSpline(theta, y, bc_type='natural')


theta_min = 0. #float(theta.min())
theta_max = np.sum(segment_lengths) #float(theta.max())

def lookup_xy(theta_query):
    return float(spline_x(theta_query)), float(spline_y(theta_query))


# ============================================================================
# METHOD 1: KD-Tree Based Lookup (RECOMMENDED - Fast and Accurate)
# ============================================================================

class ThetaLookupTable:
    """
    Pre-computed lookup table for fast position-to-theta queries
    """
    def __init__(self, spline_x, spline_y, theta_min, theta_max, n_samples=50000):
        """
        Build lookup table by densely sampling the spline
        
        Args:
            spline_x: X spline function
            spline_y: Y spline function
            theta_min: Minimum theta value
            theta_max: Maximum theta value
            n_samples: Number of samples to pre-compute (more = more accurate)
        """
        print(f"Building lookup table with {n_samples} samples...")
        
        # Dense sampling of the path
        self.theta_samples = np.linspace(theta_min, theta_max, n_samples)
        self.x_samples = spline_x(self.theta_samples)
        self.y_samples = spline_y(self.theta_samples)
        
        # Build KD-tree for fast nearest neighbor search
        self.positions = np.column_stack([self.x_samples, self.y_samples])
        self.kdtree = KDTree(self.positions)
        
        print("Lookup table built successfully!")
    
    def query(self, x_query, y_query, k_neighbors=1):
        """
        Look up theta from position
        
        Args:
            x_query: X coordinate
            y_query: Y coordinate
            k_neighbors: Number of nearest neighbors to consider
        
        Returns:
            theta: Corresponding theta value
        """
        query_point = np.array([[x_query, y_query]])
        
        if k_neighbors == 1:
            # Simple nearest neighbor
            dist, idx = self.kdtree.query(query_point)
            return float(self.theta_samples[idx[0]])
        else:
            # Average of k nearest neighbors (more robust)
            distances, indices = self.kdtree.query(query_point, k=k_neighbors)
            # Weighted average by inverse distance
            weights = 1.0 / (distances[0] + 1e-10)
            weights /= weights.sum()
            theta_weighted = np.sum(self.theta_samples[indices[0]] * weights)
            return float(theta_weighted)
    
    def query_batch(self, x_queries, y_queries, k_neighbors=1):
        """
        Look up theta for multiple positions at once (vectorized)
        
        Args:
            x_queries: Array of X coordinates
            y_queries: Array of Y coordinates
            k_neighbors: Number of nearest neighbors to consider
        
        Returns:
            thetas: Array of corresponding theta values
        """
        query_points = np.column_stack([x_queries, y_queries])
        
        if k_neighbors == 1:
            distances, indices = self.kdtree.query(query_points)
            return self.theta_samples[indices].astype(float)
        else:
            distances, indices = self.kdtree.query(query_points, k=k_neighbors)
            # Weighted average for each query point
            thetas = []
            for i in range(len(query_points)):
                weights = 1.0 / (distances[i] + 1e-10)
                weights /= weights.sum()
                theta_weighted = np.sum(self.theta_samples[indices[i]] * weights)
                thetas.append(theta_weighted)
            return np.array(thetas)
    
    def save(self, filename='theta_lookup_table.pkl'):
        """Save lookup table to file"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'theta_samples': self.theta_samples,
                'x_samples': self.x_samples,
                'y_samples': self.y_samples,
                'positions': self.positions
            }, f)
        print(f"Lookup table saved to {filename}")
    
    @classmethod
    def load(cls, filename='theta_lookup_table.pkl'):
        """Load lookup table from file"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        # Create empty instance
        instance = cls.__new__(cls)
        instance.theta_samples = data['theta_samples']
        instance.x_samples = data['x_samples']
        instance.y_samples = data['y_samples']
        instance.positions = data['positions']
        instance.kdtree = KDTree(instance.positions)
        print(f"Lookup table loaded from {filename}")
        return instance


# ============================================================================
# METHOD 2: Grid-Based Lookup (Alternative approach)
# ============================================================================

class GridBasedLookup:
    """
    Grid-based lookup table (useful for rectangular regions)
    """
    def __init__(self, spline_x, spline_y, theta_min, theta_max, n_samples=50000, grid_resolution=500):
        # Dense sampling
        theta_samples = np.linspace(theta_min, theta_max, n_samples)
        x_samples = spline_x(theta_samples)
        y_samples = spline_y(theta_samples)
        
        # Create bounding box
        self.x_min, self.x_max = x_samples.min(), x_samples.max()
        self.y_min, self.y_max = y_samples.min(), y_samples.max()
        
        # Add padding
        x_range = self.x_max - self.x_min
        y_range = self.y_max - self.y_min
        self.x_min -= 0.1 * x_range
        self.x_max += 0.1 * x_range
        self.y_min -= 0.1 * y_range
        self.y_max += 0.1 * y_range
        
        # Create grid
        self.grid_resolution = grid_resolution
        x_grid = np.linspace(self.x_min, self.x_max, grid_resolution)
        y_grid = np.linspace(self.y_min, self.y_max, grid_resolution)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        
        # Build KDTree for nearest neighbor assignment
        positions = np.column_stack([x_samples, y_samples])
        kdtree = KDTree(positions)
        
        # For each grid cell, find nearest theta
        grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
        distances, indices = kdtree.query(grid_points)
        
        self.theta_grid = theta_samples[indices].reshape(grid_resolution, grid_resolution)
        self.distance_grid = distances.reshape(grid_resolution, grid_resolution)
        
        # Store valid region (within reasonable distance to path)
        max_valid_distance = np.percentile(distances, 95)
        self.valid_mask = self.distance_grid < max_valid_distance
        
    def query(self, x_query, y_query):
        """Look up theta from grid"""
        # Convert position to grid indices
        i = int((y_query - self.y_min) / (self.y_max - self.y_min) * (self.grid_resolution - 1))
        j = int((x_query - self.x_min) / (self.x_max - self.x_min) * (self.grid_resolution - 1))
        
        # Clamp to valid range
        i = np.clip(i, 0, self.grid_resolution - 1)
        j = np.clip(j, 0, self.grid_resolution - 1)
        
        return float(self.theta_grid[i, j])


# ============================================================================
# Build and Test Lookup Table
# ============================================================================

# Build the lookup table (adjust n_samples for accuracy vs memory tradeoff)
lookup_table = ThetaLookupTable(spline_x, spline_y, theta_min, theta_max, n_samples=100000)

# Test on original waypoints
print("\nTesting lookup table accuracy on original waypoints:")
theta_recovered = []
for i, (x_query, y_query) in enumerate(zip(x, y)):
    theta_hat = lookup_table.query(x_query, y_query, k_neighbors=10)  # Use 5 neighbors for smoothing
    theta_recovered.append(theta_hat)
    
    if i < 5:
        error = abs(theta_hat - theta[i])
        print(f"Point {i}: Original θ={theta[i]:.6f}, Recovered θ={theta_hat:.6f}, Error={error:.8f}")

theta_recovered = np.array(theta_recovered)
errors = np.abs(theta_recovered - theta)
print(f"\nAverage error: {np.mean(errors):.8f}")
print(f"Max error: {np.max(errors):.8f}")
print(f"Median error: {np.median(errors):.8f}")

# Test batch query (much faster for multiple points)
print("\nTesting batch query:")
theta_batch = lookup_table.query_batch(x, y, k_neighbors=5)
print(f"Batch query completed for {len(x)} points")

# Visualize results
x_ref_ls, y_ref_ls = [], []
for theta_hat in theta_recovered:
    x_ref, y_ref = lookup_xy(theta_hat)
    x_ref_ls.append(x_ref)
    y_ref_ls.append(y_ref)

x_ref_ls = np.array(x_ref_ls)
y_ref_ls = np.array(y_ref_ls)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(x_ref_ls, y_ref_ls, 'b-', linewidth=2, label='Reconstructed path')
plt.plot(x, y, 'ro', markersize=4, label='Original waypoints', alpha=0.6)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Path Reconstruction')
plt.legend()
plt.grid(True)
plt.axis('equal')

plt.subplot(1, 3, 2)
plt.plot(theta, label='Original θ', linewidth=2)
plt.plot(theta_recovered, '--', label='Recovered θ', linewidth=2)
plt.xlabel('Waypoint index')
plt.ylabel('θ')
plt.title('Theta Recovery')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(errors)
plt.xlabel('Waypoint index')
plt.ylabel('Absolute Error')
plt.title('Recovery Error Distribution')
plt.grid(True)
plt.yscale('log')

plt.tight_layout()
plt.show()

# Save lookup table for future use
lookup_table.save('theta_lookup_table.pkl')

# Example: How to use in future sessions
# lookup_table = ThetaLookupTable.load('theta_lookup_table.pkl')
# theta = lookup_table.query(x=100.5, y=200.3)

print("\n" + "="*60)
print("USAGE EXAMPLES:")
print("="*60)
print("\n# Single query:")
print("theta = lookup_table.query(x=100.5, y=200.3)")
print("\n# Single query with smoothing (more accurate):")
print("theta = lookup_table.query(x=100.5, y=200.3, k_neighbors=5)")
print("\n# Batch query (fast for many points):")
print("thetas = lookup_table.query_batch([x1, x2, x3], [y1, y2, y3])")
print("\n# Save for later:")
print("lookup_table.save('my_table.pkl')")
print("\n# Load saved table:")
print("lookup_table = ThetaLookupTable.load('my_table.pkl')")
print("="*60)