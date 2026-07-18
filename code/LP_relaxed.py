import gurobipy as gp
from gurobipy import GRB
import itertools
import time
import sys
import ast
import math  # <-- CHANGED: Imported the math library

# --- Try to load the data ---
DATA_FILE_NAME = "adult_final_dataset.py"

try:
    with open(DATA_FILE_NAME, 'r') as f:
        file_content = f.read()
    all_points = ast.literal_eval(file_content)
    if not isinstance(all_points, list):
        raise TypeError("Data was not a list")
except FileNotFoundError:
    print(f"Error: Could not find '{DATA_FILE_NAME}'.")
    sys.exit(1)
except Exception as e:
    print(f"Error: Could not parse '{DATA_FILE_NAME}'.")
    print(f"Parser error: {e}")
    sys.exit(1)


# --- CHANGED: Now uses math.sqrt() ---
def computeDistance(point1, point2):
    """Compute Euclidean Distance between two points."""
    squared_sum = sum((point1[i] - point2[i])**2 for i in range(len(point1)))
    return math.sqrt(squared_sum)

# Build distance dictionary for all point pairs
def buildDistanceDictionary(points):
    """Build distance dictionary for all point pairs."""
    print("Building distance dictionary...")
    start_build_time = time.time()
    
    distance_dict = {}
    n = len(points)
    for i in range(n):
        for j in range(i, n):
            point_i = tuple(points[i])
            point_j = tuple(points[j])
            distance = computeDistance(point_i, point_j)
            distance_dict[(point_i, point_j)] = distance
            distance_dict[(point_j, point_i)] = distance
            
    end_build_time = time.time()
    print(f"Dictionary built in {end_build_time - start_build_time:.2f} seconds.")
    return distance_dict

# Solve LP for a given radius to test feasibility (Gurobi version)
def solve_k_center_LP(points, k, radius, distance_dict):
    """Solves the fractional K-Center LP relaxation using Gurobi."""
    n = len(points)
    point_indices = range(n)

    try:
        m = gp.Model("KCenter_LP_Feasibility")
        m.setParam('OutputFlag', 0)

        y = m.addVars(point_indices, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="y")
        x = m.addVars(point_indices, point_indices, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="x")

        for j in point_indices:
            point_j_tuple = tuple(points[j])
            feasible_centers = [i for i in point_indices 
                                if distance_dict[(tuple(points[i]), point_j_tuple)] <= radius]
            if not feasible_centers:
                m.dispose()
                return False 
            m.addConstr(gp.quicksum(x[i, j] for i in feasible_centers) >= 1, name=f"Cover_{j}")
        
        m.addConstrs((x[i, j] <= y[i] for i in point_indices for j in point_indices), name="Link")
        m.addConstr(y.sum() <= k, name="Center_Limit")

        m.setObjective(0, GRB.MINIMIZE)
        m.optimize()

        status = m.Status
        m.dispose() # Clean up model from memory

        if status == GRB.OPTIMAL:
            return True
        else:
            return False

    except gp.GurobiError as e:
        print(f"Gurobi error: {e}")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False


# --- CHANGED: Added detailed logging ---
def findMinimumFeasibleRadius_LP(points, num_disks):
    """Binary Search using LP feasibility test with detailed logs."""
    
    full_distance_dict = buildDistanceDictionary(points)
    
    relevant_distances = set()
    for i in range(len(points)):
        for j in range(i, len(points)):
             relevant_distances.add(full_distance_dict[(tuple(points[i]), tuple(points[j]))])
             
    distance_values_sorted = sorted(list(relevant_distances))
    
    if not distance_values_sorted:
        print("Error: No distances found. Do you have at least 2 points?")
        return "No distances found"
        
    print(f"Found {len(distance_values_sorted)} unique distances to test.")

    left = 0
    right = len(distance_values_sorted) - 1
    best_radius = None
    step = 1 # <-- For logging
    
    while left <= right:
        mid_idx = (left + right) // 2
        radius_candidate = distance_values_sorted[mid_idx]
        
        print(f"\n--- Binary Search Step {step} ---")
        print(f"  Search bounds: [index {left}, index {right}]")
        print(f"  Testing index {mid_idx} (Radius: {radius_candidate:.5f})")
        
        solve_start_time = time.time()
        feasible = solve_k_center_LP(points, num_disks, radius_candidate, full_distance_dict)
        solve_end_time = time.time()
        
        step_time = solve_end_time - solve_start_time
        
        if feasible:
            print(f"  ✅ Radius {radius_candidate:.5f} is FEASIBLE.")
            print(f"  (Solver time: {step_time:.2f} seconds)")
            best_radius = radius_candidate
            right = mid_idx - 1 # Try to find a smaller radius
        else:
            print(f"  ❌ Radius {radius_candidate:.5f} is NOT feasible.")
            print(f"  (Solver time: {step_time:.2f} seconds)")
            left = mid_idx + 1 # Must use a larger radius
        
        step += 1
    
    return best_radius if best_radius is not None else "No feasible radius found"


# --- Main execution block ---
if __name__ == "__main__":
    
    # --- 1. USER-CONFIGURABLE VARIABLES ---
    num_disks = 10         
    DATA_SUBSET_FRACTION = 0.03
    
    # --- 2. DATA LOADING AND SUBSETTING ---
    num_total_points = len(all_points)
    num_points_to_use = int(num_total_points * DATA_SUBSET_FRACTION)
    if num_points_to_use < 1: num_points_to_use = 1
    points = all_points[:num_points_to_use]
    
    print("\n" + "="*40)
    print("=== Running LP-based Optimal K-Center (Gurobi Version) ===")
    print(f"Loaded from '{DATA_FILE_NAME}'")
    print(f"Total points in data file: {num_total_points}")
    print(f"Using {len(points)} points ({DATA_SUBSET_FRACTION * 100}%) for this run.")
    print("="*40 + "\n")

    # --- 3. RUN THE SOLVER ---
    start_time = time.time()
    optimal_radius = findMinimumFeasibleRadius_LP(points, num_disks)
    end_time = time.time()
    
    print("\n" + "="*40)
    print(f"🌟 Optimal Radius (LP Bound) = {optimal_radius}")
    print(f"🕒 Total time taken: {end_time - start_time:.2f} seconds")
    print("="*40)