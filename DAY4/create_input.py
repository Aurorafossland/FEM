import numpy as np

# ============================================================
# Parameters (easy to change)
# ============================================================

# Domain size
Lx = 1.0   # [m]
Ly = 1.0   # [m]

# Grid spacing
dx = 0.1  # [m]
dy = 0.1  # [m]

# Exclusion zone (rectangle to remove nodes inside)
# Format: (xmin, xmax, ymin, ymax)
x_excl = (0.5, 1.0)
y_excl = (0.0, 0.5)

# Load application point
load_point = (1.0, 1.0)   # [x,y]
load_value = -0.01        # vertical load [force units]

# Output filename
outfile = "DAY4/topopt_input.m"

# ============================================================
# Node generation
# ============================================================

# Generate grid
xvals = np.arange(0, Lx + dx/2, dx)
yvals = np.arange(0, Ly + dy/2, dy)
Xgrid, Ygrid = np.meshgrid(xvals, yvals)
Xall = np.column_stack((Xgrid.ravel(), Ygrid.ravel()))



# Remove nodes inside exclusion zone
mask = ~((Xall[:, 0] > 0.5) & (Xall[:, 0] <= 1.0) & (Xall[:, 1] >= 0.0) & (Xall[:, 1] <= 0.5))
X = Xall[mask]
print(X)
nnode = X.shape[0]

def ccw(A, B, C):
    return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])

def segments_intersect(A, B, C, D):
    # Return True if line segments AB and CD intersect
    return (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D))


def is_strictly_inside_rect(x, y, x_excl, y_excl):
    return (x_excl[0] < x < x_excl[1]) and (y_excl[0] < y < y_excl[1])

def segment_crosses_rect_strict(p1, p2, x_excl, y_excl):
    # Rectangle corners
    rect = [
        (x_excl[0], y_excl[0]),
        (x_excl[1], y_excl[0]),
        (x_excl[1], y_excl[1]),
        (x_excl[0], y_excl[1])
    ]
    # Rectangle edges as segments
    rect_edges = [
        (rect[0], rect[1]),
        (rect[1], rect[2]),
        (rect[2], rect[3]),
        (rect[3], rect[0])
    ]
    # Exclude if any node is strictly inside
    if is_strictly_inside_rect(p1[0], p1[1], x_excl, y_excl) or is_strictly_inside_rect(p2[0], p2[1], x_excl, y_excl):
        return True
    # Exclude if the segment crosses the rectangle (but allow tangency)
    for q1, q2 in rect_edges:
        if segments_intersect(p1, p2, q1, q2):
            # Check if intersection is only at an endpoint (tangency or node on edge)
            # If both endpoints are on the rectangle edge, allow (tangency)
            # If intersection is not at endpoints, exclude
            # We'll use a small tolerance for floating point comparison
            tol = 1e-12
            def is_on_segment(pt, seg_a, seg_b):
                # Check if pt is one of the endpoints or lies exactly on the segment
                return (np.linalg.norm(np.array(pt) - np.array(seg_a)) < tol or
                        np.linalg.norm(np.array(pt) - np.array(seg_b)) < tol)
            # For each rectangle edge, check if p1 or p2 is on the edge
            if (is_on_segment(p1, q1, q2) or is_on_segment(p2, q1, q2)):
                continue  # Allow tangency
            else:
                return True  # Exclude if truly crossing
    return False


# ============================================================
# Connectivity (complete ground structure, excluding crossing elements)
# ============================================================

IX = []
for i in range(nnode):
    for j in range(i + 1, nnode):
        p1 = X[i]
        p2 = X[j]
        if not segment_crosses_rect_strict(p1, p2, x_excl, y_excl):
            IX.append([i + 1, j + 1, 1])  # MATLAB 1-based indexing
IX = np.array(IX, dtype=int)


# ============================================================
# Properties
# ============================================================
mprop = np.array([[1.0, 1.0]])

# ============================================================
# Loads
# ============================================================
# Find nearest node to load_point
loadNode = np.argmin(np.sum((X - np.array(load_point))**2, axis=1)) + 1
loads = np.array([[loadNode, 2, load_value]])

# ============================================================
# Boundary conditions
# ============================================================
bound = []
tol = 1e-8
for i, (x, y) in enumerate(X, start=1):
    if abs(y) < tol:  # nodes on x-axis
        bound.append([i, 2, 0.0])
# Fix rigid body motion at node (0,0)
fixNode = np.argmin(np.sum((X - np.array([0, 0]))**2, axis=1)) + 1
bound.append([fixNode, 1, 0.0])
bound.append([fixNode, 2, 0.0])
bound = np.array(bound)

# ============================================================
# Control parameter
# ============================================================
plotdof = loadNode

# ============================================================
# Volume constraint
# ============================================================
V_cond = 6

# ============================================================
# Maximum number of iterations
# ============================================================
iot_max = 50

# ============================================================
# Write MATLAB file
# ============================================================
with open(outfile, "w") as f:
    f.write("clear all\n\n")
    f.write("%% Coordinates of nodes\n")
    f.write("X = [\n")
    for row in X:
        f.write(f"  {row[0]:.2f}  {row[1]:.2f}\n")
    f.write("];\n\n")

    f.write("%% Topology matrix IX(node1,node2,propno)\n")
    f.write("IX = [\n")
    for row in IX:
        f.write(f"  {row[0]}  {row[1]}  {row[2]}\n")
    f.write("];\n\n")

    f.write("%% Element property matrix mprop = [ E A ]\n")
    f.write("mprop = [ 1.0 1.0 ];\n\n")

    f.write("%% Prescribed loads mat(node,ldof,force)\n")
    f.write("loads = [\n")
    for row in loads:
        f.write(f"  {int(row[0])}  {int(row[1])}  {row[2]:.4f}\n")
    f.write("];\n\n")

    f.write("%% Boundary conditions mat(node,ldof,disp)\n")
    f.write("bound = [\n")
    f.write(f"  {1}  {1}  {0.0}\n")
    for row in bound:
        f.write(f"  {int(row[0])}  {int(row[1])}  {row[2]:.1f}\n")
    f.write("];\n\n")

    f.write("%% Control Parameters\n")
    f.write(f"plotdof = {plotdof};\n")

    f.write(f"V_cond = {V_cond};\n")
    f.write(f"max_iopt = {iot_max};\n")

print(f"MATLAB input file '{outfile}' generated.")
