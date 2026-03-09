### Question:
Can you create a jupyter notebook for this app for learning purposes? What i want is a jupyter notebook that will gradually explain how as rigid as possible deformation works

### Answer:
Great idea. I’ll add a new Jupyter notebook that walks through the core ideas behind As-Rigid-As-Possible (ARAP) 2D deformation, using simplified, self-contained examples. It won’t depend on your app’s internal classes, so you can run it anywhere to learn the math, and then map the concepts back to your project.

Below is a suggested notebook file you can save as ARAP_2D_Introduction.ipynb at the project root. It’s structured as a step-by-step tutorial:
- Part 1: Representing 2D meshes (triangles)
- Part 2: Local step (best-fitting rotation)
- Part 3: Global step (solving for updated vertex positions)
- Part 4: Constraints/handles
- Part 5: Interactive experiment (simple draggable handles inside the notebook)
- Part 6: Relation to the codebase (how these ideas map to your src/rigid_mesh_deformer.py)

Note: This is a JSON .ipynb file. Create it with this exact content.

```json
// File: D:/ProgrammingPython/ZGL Projects/deform2d/ARAP_2D_Introduction.ipynb
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# As-Rigid-As-Possible (ARAP) Deformation in 2D\n",
    "\n",
    "This notebook is a step-by-step, practical introduction to ARAP surface deformation for 2D triangle meshes. You'll:\n",
    "- Build a tiny 2D mesh\n",
    "- Understand the ARAP energy: local and global steps\n",
    "- Compute per-triangle best-fitting rotations\n",
    "- Assemble and solve the global linear system with positional constraints (handles)\n",
    "- Experiment with dragging handles and see the deformation\n",
    "\n",
    "We'll keep the math minimal but precise, and provide runnable NumPy code you can extend. At the end, we map the concepts to your project code in `src/rigid_mesh_deformer.py`."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 0. Imports and Utilities\n",
    "We use NumPy for linear algebra and Matplotlib for visualization."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from matplotlib.collections import LineCollection\n",
    "from dataclasses import dataclass\n",
    "\n",
    "np.set_printoptions(suppress=True, precision=4)\n",
    "\n",
    "def vec2(v):\n",
    "    v = np.asarray(v, dtype=np.float64)\n",
    "    return v.reshape(2,)\n",
    "\n",
    "def normalize2(v):\n",
    "    n = np.linalg.norm(v)\n",
    "    return v/n if n > 1e-12 else v\n",
    "\n",
    "def edges_from_faces(F):\n",
    "    # F: (M,3) triangles, returns unique undirected edges\n",
    "    E = set()\n",
    "    for (a,b,c) in F:\n", 
    "        for u, v in [(a,b),(b,c),(c,a)]:\n",
    "            E.add(tuple(sorted((int(u), int(v)))))\n",
    "    return np.array(sorted(list(E)), dtype=int)\n",
    "\n",
    "def cot(a, b, c):\n",
    "    # cotangent at vertex a of triangle (a,b,c)\n",
    "    u = b - a\n",
    "    v = c - a\n",
    "    cos_ = u @ v\n",
    "    sin_ = np.cross(np.append(u,0), np.append(v,0))[2]\n",
    "    return cos_/sin_ if abs(sin_) > 1e-12 else 0.0"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. A Tiny 2D Mesh\n",
    "We'll start with a simple square domain triangulated into two triangles. We'll also compute the cotangent Laplacian (common in ARAP implementations) for the global step."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Square with two triangles\n",
    "V0 = np.array([\n",
    "    [0.0, 0.0],  # 0\n",
    "    [1.0, 0.0],  # 1\n",
    "    [1.0, 1.0],  # 2\n",
    "    [0.0, 1.0],  # 3\n",
    "], dtype=np.float64)\n",
    "\n",
    "F = np.array([\n",
    "    [0, 1, 2],\n",
    "    [0, 2, 3]\n",
    "], dtype=int)\n",
    "\n",
    "V = V0.copy()  # deformed positions start equal to rest positions\n",
    "\n",
    "def plot_mesh(V, F, ax=None, color='k', lw=2, pts=True, title=None):\n",
    "    if ax is None:\n",
    "        fig, ax = plt.subplots()\n",
    "    segments = []\n",
    "    for (a,b,c) in F:\n",
    "        segments += [ [V[a], V[b]], [V[b], V[c]], [V[c], V[a]] ]\n",
    "    lc = LineCollection(segments, colors=color, linewidths=lw)\n",
    "    ax.add_collection(lc)\n",
    "    if pts:\n",
    "        ax.scatter(V[:,0], V[:,1], color=color)\n",
    "    ax.set_aspect('equal')\n",
    "    ax.set_xlim(-0.2, 1.2)\n",
    "    ax.set_ylim(-0.2, 1.2)\n",
    "    if title:\n",
    "        ax.set_title(title)\n",
    "    return ax\n",
    "\n",
    "ax = plot_mesh(V, F, title='Initial Mesh (rest = deformed)')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. ARAP Energy Overview\n",
    "ARAP seeks new vertex positions V minimizing an energy that tries to preserve local rigidity while matching user constraints. For each triangle t with vertices i,j,k:\n",
    "- Local step: find rotation R_t that best aligns current deformed edges to rest edges\n",
    "- Global step: solve a sparse linear system that updates vertex positions to be as close as possible to the rotated rest edges while respecting constraints\n",
    "\n",
    "We iterate Local → Global until convergence (or a few iterations suffice)."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Local Step: Best-Fitting Rotation per Triangle\n",
    "We want R_t ∈ SO(2) minimizing || (V_i - V_j) - R_t (V0_i - V0_j) ||^2 over triangle edges. In practice, we can compute R_t via a 2x2 SVD of a covariance matrix:\n",
    "C_t = Σ_e w_e (v_e)(v0_e)^T, where v_e are current deformed edges and v0_e are rest edges, with weights w_e (cotangent weights or uniform).\n",
    "\n",
    "Then R_t = U diag(1, det(UV^T)) V^T, where C_t = U S V^T (SVD)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def triangle_edges(V, f):\n",
    "    i, j, k = f\n",
    "    return (V[j]-V[i], V[k]-V[j], V[i]-V[k])  # oriented edges across triangle\n",
    "\n", 
    "def best_rotation_2x2(C):\n",
    "    U, S, Vt = np.linalg.svd(C)\n",
    "    R = U @ Vt\n",
    "    # ensure det(R)=+1\n",
    "    if np.linalg.det(R) < 0:\n",
    "        U[:,1] *= -1\n",
    "        R = U @ Vt\n",
    "    return R\n",
    "\n",
    "def local_step(V, V0, F, edge_weight='uniform'):\n",
    "    # returns list of 2x2 rotations per triangle\n",
    "    R_list = []\n",
    "    for f in F:\n",
    "        e0 = triangle_edges(V0, f)\n",
    "        e  = triangle_edges(V,  f)\n",
    "        if edge_weight == 'uniform':\n",
    "            w = [1.0, 1.0, 1.0]\n",
    "        else:\n",
    "            # simple cot weights per corner mapped to opposite edge\n",
    "            i,j,k = f\n",
    "            wi = max(cot(V0[i], V0[j], V0[k]), 0.0)\n",
    "            wj = max(cot(V0[j], V0[k], V0[i]), 0.0)\n",
    "            wk = max(cot(V0[k], V0[i], V0[j]), 0.0)\n",
    "            w = [wi, wj, wk]\n",
    "        C = np.zeros((2,2))\n",
    "        for we, ve, v0e in zip(w, e, e0):\n",
    "            C += we * np.outer(ve, v0e)\n",
    "        R = best_rotation_2x2(C)\n",
    "        R_list.append(R)\n",
    "    return R_list\n",
    "\n",
    "R_list = local_step(V, V0, F, edge_weight='uniform')\n",
    "R_list"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Initially, V == V0 so the best rotations are the identity (or numerically close)."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Global Step: Solve for Vertex Positions\n",
    "Given R_t, we solve for V by minimizing sum over triangles and their edges of w_e || (V_i - V_j) - R_t (V0_i - V0_j) ||^2, subject to position constraints.\n",
    "\n",
    "This yields a sparse linear system: L V = B, where L is a Laplacian-like matrix and B encodes rotated rest edges. Constraints are applied by pinning rows or using soft constraints (we'll use hard constraints/pinning for clarity)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def build_uniform_laplacian(n, F):\n",
    "    # simple graph Laplacian (uniform weights)\n",
    "    E = edges_from_faces(F)\n",
    "    L = np.zeros((n,n))\n",
    "    for a,b in E:\n",
    "        L[a,a] += 1; L[b,b] += 1\n",
    "        L[a,b] -= 1; L[b,a] -= 1\n",
    "    return L\n",
    "\n",
    "def build_rhs(V0, F, R_list):\n",
    "    # Each triangle contributes to B via rotated rest edges.\n",
    "    # For uniform Laplacian, a common derivation yields B = sum over neighbors of rotated differences.\n",
    "    n = V0.shape[0]\n",
    "    B = np.zeros((n,2))\n",
    "    # Accumulate contributions per triangle\n",
    "    for t, f in enumerate(F):\n",
    "        i,j,k = f\n",
    "        R = R_list[t]\n",
    "        # triangle edges in rest pose\n",
    "        e_ij0 = V0[i]-V0[j]; e_jk0 = V0[j]-V0[k]; e_ki0 = V0[k]-V0[i]\n",
    "        # rotated edges\n",
    "        r_ij = R @ e_ij0; r_jk = R @ e_jk0; r_ki = R @ e_ki0\n",
    "        # Each edge contributes to two vertices with opposite signs (uniform weights)\n",
    "        B[i] += r_ij - r_ki\n",
    "        B[j] += r_jk - r_ij\n",
    "        B[k] += r_ki - r_jk\n",
    "    return B\n",
    "\n",
    "def apply_constraints(L, B, constraints):\n",
    "    # constraints: dict {vertex_index: target_xy}\n",
    "    Lc = L.copy()\n",
    "    Bc = B.copy()\n",
    "    for idx, pos in constraints.items():\n",
    "        # zero row\n",
    "        Lc[idx,:] = 0\n",
    "        Lc[idx,idx] = 1\n",
    "        # set rhs to target\n",
    "        Bc[idx,:] = pos\n",
    "    return Lc, Bc\n",
    "\n",
    "def global_step(V, V0, F, R_list, constraints):\n",
    "    n = V.shape[0]\n",
    "    L = build_uniform_laplacian(n, F)\n",
    "    B = build_rhs(V0, F, R_list)\n",
    "    Lc, Bc = apply_constraints(L, B, constraints)\n",
    "    # Solve two independent systems for x and y\n",
    "    X = np.linalg.solve(Lc, Bc[:,0])\n",
    "    Y = np.linalg.solve(Lc, Bc[:,1])\n",
    "    return np.column_stack((X,Y))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. One ARAP Iteration\n",
    "Now we can run Local → Global once. Let’s add constraints: pin the bottom-left vertex (0) and drag the top-right (2)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "constraints = {\n",
    "    0: V0[0],      # pin\n",
    "    2: V0[2] + np.array([0.3, 0.0])  # drag to the right\n",
    "}\n",
    "\n",
    "R_list = local_step(V, V0, F)\n",
    "V1 = global_step(V, V0, F, R_list, constraints)\n",
    "\n",
    "fig, ax = plt.subplots(1,2, figsize=(10,4))\n",
    "plot_mesh(V0, F, ax=ax[0], color='gray', title='Rest Mesh')\n",
    "plot_mesh(V1, F, ax=ax[1], color='C0', title='After 1 ARAP Iteration')\n",
    "ax[1].scatter([constraints[2][0]],[constraints[2][1]], color='red', s=60, label='Handle target')\n",
    "ax[1].legend()\n",
    "plt.show()\n",
    "\n",
    "V = V1.copy()  # update current positions"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Iterate a Few Times\n",
    "Usually 5–10 iterations are enough for a stable result."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def arap_solve(V0, F, constraints, iters=5):\n",
    "    V = V0.copy()\n",
    "    for _ in range(iters):\n",
    "        R_list = local_step(V, V0, F)\n",
    "        V = global_step(V, V0, F, R_list, constraints)\n",
    "    return V\n",
    "\n",
    "V_arp = arap_solve(V0, F, constraints, iters=10)\n",
    "ax = plot_mesh(V0, F, color='gray', title='ARAP Result (10 iters)')\n",
    "plot_mesh(V_arp, F, ax=ax, color='C1')\n",
    "ax.scatter([constraints[2][0]],[constraints[2][1]], color='red', s=60)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Interactive Experiment (Drag a Handle)\n",
    "Use the widget below to move vertex 2 around and see the deformation update. This is a minimal in-notebook analogue to your PyQt app's interactive dragging."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "%matplotlib notebook\n",
    "\n",
    "V_current = V0.copy()\n",
    "handle_idx = 2\n",
    "fixed_idx = 0\n",
    "\n",
    "fig, ax = plt.subplots()\n",
    "plot_mesh(V0, F, ax=ax, color='gray', title='Drag the red handle')\n",
    "mesh_line = plot_mesh(V_current, F, ax=ax, color='C0')\n",
    "handle_scatter = ax.scatter([V_current[handle_idx,0]],[V_current[handle_idx,1]], color='red', s=60)\n",
    "dragging = {'active': False}\n",
    "\n",
    "def redraw():\n",
    "    ax.cla()\n",
    "    plot_mesh(V0, F, ax=ax, color='gray')\n",
    "    plot_mesh(V_current, F, ax=ax, color='C0')\n",
    "    ax.scatter([V_current[handle_idx,0]],[V_current[handle_idx,1]], color='red', s=60)\n",
    "    fig.canvas.draw_idle()\n",
    "\n",
    "def on_press(event):\n",
    "    if event.inaxes != ax: return\n",
    "    p = np.array([event.xdata, event.ydata])\n",
    "    if np.linalg.norm(p - V_current[handle_idx]) < 0.1:\n",
    "        dragging['active'] = True\n",
    "\n",
    "def on_release(event):\n",
    "    dragging['active'] = False\n",
    "\n",
    "def on_move(event):\n",
    "    if not dragging['active'] or event.inaxes != ax:\n",
    "        return\n",
    "    p = np.array([event.xdata, event.ydata])\n",
    "    constraints = {fixed_idx: V0[fixed_idx], handle_idx: p}\n",
    "    V_new = arap_solve(V0, F, constraints, iters=5)\n",
    "    V_current[:] = V_new\n",
    "    redraw()\n",
    "\n",
    "cid1 = fig.canvas.mpl_connect('button_press_event', on_press)\n",
    "cid2 = fig.canvas.mpl_connect('button_release_event', on_release)\n",
    "cid3 = fig.canvas.mpl_connect('motion_notify_event', on_move)\n",
    "redraw()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Tip: If the interactive plot doesn't work in your environment, switch to `%matplotlib qt` or `%matplotlib widget`, or run in JupyterLab with ipympl installed."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 8. Weights, Stability, and Variants\n",
    "- Uniform vs. cotangent weights: cotangent weights (derived from mesh geometry) generally give better shape preservation.\n",
    "- Hard vs. soft constraints: we pin rows here. Soft constraints can be handled by adding large weights on constraint rows.\n",
    "- 2D vs. 3D: The rotation step generalizes via 2x2 vs. 3x3 SVD; the global system structure is similar.\n",
    "- Pre-factorization: In interactive settings, the system matrix with pinned constraints can be factored once and reused; only the RHS changes as handles move (your app does this via LU factorization in SciPy)."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 9. Mapping to Your Codebase\n",
    "In your project:\n",
    "- `src/rigid_mesh_deformer.py` holds the core ARAP logic. Look for methods analogous to:\n",
    "  - Local step: computing per-triangle orientation (rotation) matrices (e.g., `precompute_orientation_matrix`, `update_scaled_triangle`) and the fitting step.\n",
    "  - Global step: assembling the linear system and solving with LU (`precompute_fitting_matrices`, `apply_fitting_step`, `validate_deformed_mesh`).\n",
    "- `src/deform_gl_widget.py` coordinates the mesh, constraints (handles), and calls `update_deformed_mesh` when dragging.\n",
    "\n",
    "Key parallels with this notebook:\n",
    "- Our `local_step` ~ your per-triangle rotation estimation using rest vs. current edges.\n",
    "- Our `global_step` with constraints ~ your linear solve using a Laplacian-like matrix and constraints.\n",
    "- Iteration loop ~ your repeated validation/update cycle during interactive dragging.\n",
    "\n",
    "This notebook purposefully uses a small mesh and a uniform Laplacian for clarity. Your application uses more robust formulations and precomputations optimized for interactivity and arbitrary meshes."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 10. Next Steps\n",
    "- Replace the simple square with a more complex 2D mesh (e.g., load a .obj and keep only XY).\n",
    "- Switch to cotangent weights in both local and global steps for improved behavior.\n",
    "- Implement pre-factorization of the system matrix for speed with moving handles.\n",
    "- Extend to multiple handles and constraints.\n",
    "- Compare results visually with your PyQt app by exporting intermediate states."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
```

How to use:
- Save the file at D:/ProgrammingPython/ZGL Projects/deform2d/ARAP_2D_Introduction.ipynb
- Open it with Jupyter:
```
jupyter notebook "D:/ProgrammingPython/ZGL Projects/deform2d/ARAP_2D_Introduction.ipynb"
```
- Run cells in order. For the interactive section, you may need to tweak the Matplotlib backend (e.g., %matplotlib widget) depending on your environment.

If you’d like, I can also add a short section that loads your assets/man.obj, flattens to 2D, and runs the same ARAP loop for a denser mesh.

### Question:
Can you use also show the assets/man.obj run the same arap loop?

### Answer:
Absolutely. Below are minimal additions to your existing notebook to load assets/man.obj, extract its XY mesh, build a simple triangulation, and run the same ARAP loop with a couple of handles.

Notes:
- Wavefront OBJ files can contain arbitrary polygon faces. To keep this educational and dependency-light, we’ll:
  - Read only vertices (v) and faces (f) from assets/man.obj.
  - Triangulate faces that have more than 3 vertices by a simple fan triangulation (sufficient for many meshes; for complex non-convex polygons you might prefer a robust triangulator).
- We’ll pick two handles heuristically: the left-most and right-most vertices in XY, pin the left-most, and drag the right-most.

Please append the following cells at the end of your notebook.

```python
# File: D:/ProgrammingPython/ZGL Projects/deform2d/arap.ipynb
#%% md
## 11. Running ARAP on assets/man.obj (XY)
We'll now load the OBJ at `assets/man.obj`, extract its XY coordinates, triangulate faces (fan triangulation for polygons), and run the same ARAP loop. We'll choose two handle vertices automatically:
- Fixed: left-most vertex
- Dragged: right-most vertex (moved slightly to the right)
#%%
import os

def read_obj_xy_tris(path):
    """
    Lightweight OBJ reader for 2D use:
    - Reads 'v' lines (keeps x,y; ignores z)
    - Reads 'f' lines and triangulates polygons into triangle fans
    Returns:
      Vxy: (n, 2) float64
      F:   (m, 3) int
    """
    V = []
    F = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if parts[0] == 'v':
                # vertex: v x y [z]
                x = float(parts[1]); y = float(parts[2])
                V.append([x, y])
            elif parts[0] == 'f':
                # face: f i[/..] j[/..] k[/..] ...
                idxs = []
                for t in parts[1:]:
                    s = t.split('/')[0]
                    idxs.append(int(s) - 1)  # OBJ is 1-based
                # fan triangulation if polygon
                if len(idxs) >= 3:
                    for k in range(1, len(idxs)-1):
                        F.append([idxs[0], idxs[k], idxs[k+1]])
    if len(V) == 0 or len(F) == 0:
        raise ValueError("OBJ did not contain vertices or faces.")
    return np.array(V, dtype=np.float64), np.array(F, dtype=int)

obj_path = os.path.join('assets', 'man.obj')
V0_man, F_man = read_obj_xy_tris(obj_path)
print(f"Loaded mesh: {V0_man.shape[0]} vertices, {F_man.shape[0]} triangles")

ax = plot_mesh(V0_man, F_man, color='gray', title='assets/man.obj (XY)')
plt.show()
#%% md
### Choose handles (left-most pinned, right-most dragged)
#%%
# Identify left-most and right-most vertices in XY
left_idx  = int(np.argmin(V0_man[:,0]))
right_idx = int(np.argmax(V0_man[:,0]))

# Create a small horizontal displacement for the right-most vertex
drag_delta = np.array([0.1*(V0_man[:,0].max()-V0_man[:,0].min()+1e-6), 0.0])
constraints_man = {
    left_idx: V0_man[left_idx],
    right_idx: V0_man[right_idx] + drag_delta
}

print("Fixed handle (left-most) index:", left_idx, "pos:", V0_man[left_idx])
print("Dragged handle (right-most) index:", right_idx, "target:", constraints_man[right_idx])

# Run a few ARAP iterations
V_man_deformed = arap_solve(V0_man, F_man, constraints_man, iters=10)

fig, ax = plt.subplots(1,2, figsize=(12,5))
plot_mesh(V0_man, F_man, ax=ax[0], color='gray', title='Rest (man.obj)')
ax[0].scatter([V0_man[left_idx,0]],[V0_man[left_idx,1]], color='green', s=40, label='fixed')
ax[0].scatter([V0_man[right_idx,0]],[V0_man[right_idx,1]], color='orange', s=40, label='dragged (rest)')
ax[0].legend()

plot_mesh(V_man_deformed, F_man, ax=ax[1], color='C1', title='ARAP Deformed (10 iters)')
ax[1].scatter([constraints_man[right_idx][0]],[constraints_man[right_idx][1]], color='red', s=60, label='handle target')
ax[1].legend()
plt.show()
#%% md
### Optional: Interactive dragging on man.obj
Drag the right-most vertex target and re-solve ARAP (runs a few iterations per move). On very large meshes, this can be slow without prefactorization.
#%%
%matplotlib notebook

V_current_man = V0_man.copy()
fixed_idx = left_idx
handle_idx = right_idx

fig, ax = plt.subplots()
plot_mesh(V0_man, F_man, ax=ax, color='gray', title='Drag the orange handle target')
plot_mesh(V_current_man, F_man, ax=ax, color='C0')
target = V0_man[handle_idx] + drag_delta
handle_target = ax.scatter([target[0]],[target[1]], color='orange', s=60)
fixed_pt = ax.scatter([V0_man[fixed_idx,0]],[V0_man[fixed_idx,1]], color='green', s=40)
dragging = {'active': False}

def redraw_man():
    ax.cla()
    plot_mesh(V0_man, F_man, ax=ax, color='gray')
    plot_mesh(V_current_man, F_man, ax=ax, color='C0')
    ax.scatter([V0_man[fixed_idx,0]],[V0_man[fixed_idx,1]], color='green', s=40)
    ax.scatter([target[0]],[target[1]], color='orange', s=60)
    fig.canvas.draw_idle()

def on_press_man(event):
    if event.inaxes != ax: return
    p = np.array([event.xdata, event.ydata])
    if np.linalg.norm(p - target) < 0.05:
        dragging['active'] = True

def on_release_man(event):
    dragging['active'] = False

def on_move_man(event):
    global target
    if not dragging['active'] or event.inaxes != ax:
        return
    target = np.array([event.xdata, event.ydata])
    cons = {fixed_idx: V0_man[fixed_idx], handle_idx: target}
    V_new = arap_solve(V0_man, F_man, cons, iters=5)
    V_current_man[:] = V_new
    redraw_man()

cid1 = fig.canvas.mpl_connect('button_press_event', on_press_man)
cid2 = fig.canvas.mpl_connect('button_release_event', on_release_man)
cid3 = fig.canvas.mpl_connect('motion_notify_event', on_move_man)
redraw_man()
```

### Question:
On the interactive dragging, can you color the vertex that I should drag? also, what is the use of %matplotlib notebook

### Answer:
Great questions.

1) Color the vertex to drag
In your man.obj interactive cell, you’re already plotting an orange “target” point that you drag. To make it more obvious which vertex is the handle, also draw the actual handle vertex (the current position of handle_idx in the deformed mesh) in a distinct color, e.g., red, and label it in the title.

Apply these minimal changes:

- Add a red scatter for the current handle vertex in the initial draw.
- Update redraw_man() to keep drawing that red handle vertex.
- Slightly clarify the title.

Here are the edits to your existing cells.

```python
# File: D:/ProgrammingPython/ZGL Projects/deform2d/arap.ipynb
# ... existing code above unchanged

fig, ax = plt.subplots()
plot_mesh(V0_man, F_man, ax=ax, color='gray', title='Drag the orange handle target (red = current handle vertex)')
plot_mesh(V_current_man, F_man, ax=ax, color='C0')
target = V0_man[handle_idx] + drag_delta
handle_target = ax.scatter([target[0]],[target[1]], color='orange', s=60, label='handle target')
fixed_pt = ax.scatter([V0_man[fixed_idx,0]],[V0_man[fixed_idx,1]], color='green', s=40, label='fixed')
# draw the current handle vertex position in red
handle_vertex_scatter = ax.scatter([V_current_man[handle_idx,0]],[V_current_man[handle_idx,1]], color='red', s=50, label='current handle vertex')
ax.legend(loc='best')
dragging = {'active': False}

def redraw_man():
    ax.cla()
    plot_mesh(V0_man, F_man, ax=ax, color='gray', title='Drag the orange handle target (red = current handle vertex)')
    plot_mesh(V_current_man, F_man, ax=ax, color='C0')
    # fixed point (green)
    ax.scatter([V0_man[fixed_idx,0]],[V0_man[fixed_idx,1]], color='green', s=40, label='fixed')
    # handle target (orange) — where you drag to
    ax.scatter([target[0]],[target[1]], color='orange', s=60, label='handle target')
    # current handle vertex (red) — current deformed position of that vertex
    ax.scatter([V_current_man[handle_idx,0]],[V_current_man[handle_idx,1]], color='red', s=50, label='current handle vertex')
    ax.legend(loc='best')
    fig.canvas.draw_idle()
```

This keeps:
- Green dot = fixed vertex (pinned)
- Orange dot = target you drag
- Red dot = the actual mesh vertex that moves because of ARAP

2) What is the use of %matplotlib notebook?
- It’s a Jupyter “magic” command that switches Matplotlib to an interactive, inline backend (the Notebook backend).
- With %matplotlib notebook, plots are rendered inside the notebook but support interactive features: mouse events, dynamic updates, zoom toolbar, and animations without spawning external windows.
- Your drag handlers rely on Matplotlib mouse events and redraws. Using %matplotlib notebook ensures those events work inline.
- Alternatives:
  - %matplotlib widget: Uses ipympl; provides richer interactivity in JupyterLab/Notebook (recommended if available).
  - %matplotlib qt: Opens figures in external Qt windows (useful if inline interaction is problematic).
  - %matplotlib inline: Static images only; no interactivity. Not suitable for the drag demo.

### Question:
Show how to switch to cotangent weights to improve behavior

### Answer:
Below are minimal changes you can drop into your existing notebook to switch the ARAP formulation from uniform weights to cotangent weights. This improves rigidity preservation and reduces shearing on irregular meshes like assets/man.obj.

What changes:
- Add a cotangent Laplacian builder.
- Add a cotangent-weighted RHS builder consistent with the ARAP derivation.
- Update local_step to use cotangent weights.
- Provide a global_step_cot that uses the cotangent Laplacian and RHS.
- Show how to run ARAP with cotangent weights on both the square and man.obj.

1) Add adjacency helpers and cotangent Laplacian

```python
# File: D:/ProgrammingPython/ZGL Projects/deform2d/arap.ipynb
#%%

def vertex_face_adjacency(n, F):
    vf = [[] for _ in range(n)]
    for t, (i,j,k) in enumerate(F):
        vf[i].append(t); vf[j].append(t); vf[k].append(t)
    return vf

def cotangent_laplacian(V0, F):
    """
    Symmetric cotangent Laplacian L (n x n).
    L_ij = -0.5 * (cot(alpha_ij) + cot(beta_ij)) for edge ij,
    L_ii = -sum_{j != i} L_ij
    """
    n = V0.shape[0]
    L = np.zeros((n, n), dtype=np.float64)
    for (i, j, k) in F:
        # For triangle (i,j,k), angles at i,j,k influence opposite edges
        # cot at i for angle between (j-i) and (k-i)
        cot_i = cot(V0[i], V0[j], V0[k])
        cot_j = cot(V0[j], V0[k], V0[i])
        cot_k = cot(V0[k], V0[i], V0[j])

        # Add contributions: for edge (j,k), add cot at i; etc.
        # Edge (j,k) gets cot_i
        for a, b, w in [(j, k, cot_i), (k, i, cot_j), (i, j, cot_k)]:
            L[a, b] -= 0.5 * w
            L[b, a] -= 0.5 * w
            L[a, a] += 0.5 * w
            L[b, b] += 0.5 * w
    return L
```

2) Cotangent-weighted RHS for ARAP

The cotangent-weighted ARAP global system can be written as L V = B, with L as the cotangent Laplacian. The RHS B sums, for each edge (i,j) in triangle t, the cotangent weight times the rotated rest edge (v0_i - v0_j). One consistent way:

```python
# File: D:/ProgrammingPython/ZGL Projects/deform2d/arap.ipynb
#%%

def build_rhs_cot(V0, F, R_list):
    """
    Cotangent-weighted RHS consistent with ARAP:
    For each triangle t=(i,j,k) with rotation R_t, edges are (i,j), (j,k), (k,i).
    Each edge contributes w_ij * R_t * (x_i^0 - x_j^0) to vertex i
    and w_ij * R_t * (x_j^0 - x_i^0) to vertex j, where w_ij is
    the cotangent weight for edge (i,j) in this triangle (from the angle opposite that edge).
    We accumulate contributions from all incident triangles.
    """
    n = V0.shape[0]
    B = np.zeros((n, 2), dtype=np.float64)

    for t, (i, j, k) in enumerate(F):
        R = R_list[t]

        # Opposite-angle cot weights per edge
        w_ij = max(cot(V0[k], V0[i], V0[j]), 0.0)  # angle at k opposite to edge (i,j)
        w_jk = max(cot(V0[i], V0[j], V0[k]), 0.0)  # angle at i opposite to edge (j,k)
        w_ki = max(cot(V0[j], V0[k], V0[i]), 0.0)  # angle at j opposite to edge (k,i)

        # Rest-space edges
        e_ij0 = V0[i] - V0[j]
        e_jk0 = V0[j] - V0[k]
        e_ki0 = V0[k] - V0[i]

        # Rotated edges
        r_ij = R @ e_ij0
        r_jk = R @ e_jk0
        r_ki = R @ e_ki0

        # Accumulate symmetric contributions
        B[i] += w_ij * r_ij - w_ki * r_ki
        B[j] += w_jk * r_jk - w_ij * r_ij
        B[k] += w_ki * r_ki - w_jk * r_jk

    return B
```

3) Local step with cotangent weights

Swap the local rotation estimate to use cotangent weights by passing edge_weight='cot' (we already added this option in local_step). If not present, add the branch as shown below:

```python
# File: D:/ProgrammingPython/ZGL Projects/deform2d/arap.ipynb
#%%

# local_step already contains edge_weight parameter. Ensure the 'cot' branch exists:
# if edge_weight == 'uniform': ...
# else: uses wi, wj, wk from cot(...) calls
# Nothing else to change here if your notebook already has the 'else' that computes cot weights.
```

4) Constraint application stays the same

We can reuse apply_constraints.

5) Global step (cotangent)

Provide a separate function so you can easily switch between uniform and cot versions:

```python
# File: D:/ProgrammingPython/ZGL Projects/deform2d/arap.ipynb
#%%

def global_step_cot(V, V0, F, R_list, constraints):
    L = cotangent_laplacian(V0, F)
    B = build_rhs_cot(V0, F, R_list)
    Lc, Bc = apply_constraints(L, B, constraints)
    X = np.linalg.solve(Lc, Bc[:, 0])
    Y = np.linalg.solve(Lc, Bc[:, 1])
    return np.column_stack((X, Y))
```

6) A convenience ARAP solver using cotangent weights

```python
# File: D:/ProgrammingPython/ZGL Projects/deform2d/arap.ipynb
#%%

def arap_solve_cot(V0, F, constraints, iters=5):
    V = V0.copy()
    for _ in range(iters):
        R_list = local_step(V, V0, F, edge_weight='cot')
        V = global_step_cot(V, V0, F, R_list, constraints)
    return V
```

7) Use it on the square mesh and man.obj

Example for the square:

```python
# File: D:/ProgrammingPython/ZGL Projects/deform2d/arap.ipynb
#%%

constraints_square = {0: V0[0], 2: V0[2] + np.array([0.3, 0.0])}
V_arp_cot = arap_solve_cot(V0, F, constraints_square, iters=10)

ax = plot_mesh(V0, F, color='gray', title='ARAP (cotangent weights) — Square')
plot_mesh(V_arp_cot, F, ax=ax, color='C2')
ax.scatter([constraints_square[2][0]], [constraints_square[2][1]], color='red', s=60)
plt.show()
```

And for man.obj (reusing left_idx, right_idx, drag_delta defined earlier):

```python
# File: D:/ProgrammingPython/ZGL Projects/deform2d/arap.ipynb
#%%

constraints_man_cot = {
    left_idx: V0_man[left_idx],
    right_idx: V0_man[right_idx] + drag_delta
}
V_man_deformed_cot = arap_solve_cot(V0_man, F_man, constraints_man_cot, iters=10)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
plot_mesh(V0_man, F_man, ax=ax[0], color='gray', title='Rest (man.obj)')
ax[0].scatter([V0_man[left_idx, 0]], [V0_man[left_idx, 1]], color='green', s=40, label='fixed')
ax[0].scatter([V0_man[right_idx, 0]], [V0_man[right_idx, 1]], color='orange', s=40, label='dragged (rest)')
ax[0].legend()

plot_mesh(V_man_deformed_cot, F_man, ax=ax[1], color='C2', title='ARAP (cot weights) Deformed (10 iters)')
ax[1].scatter([constraints_man_cot[right_idx][0]],[constraints_man_cot[right_idx][1]], color='red', s=60, label='handle target')
ax[1].legend()
plt.show()
```

8) Optional: make the interactive demo use cotangent weights

Replace arap_solve with arap_solve_cot in the motion handler:

```python
# File: D:/ProgrammingPython/ZGL Projects/deform2d/arap.ipynb
#%%

def on_move_man(event):
    global target
    if not dragging['active'] or event.inaxes != ax:
        return
    target = np.array([event.xdata, event.ydata])
    cons = {fixed_idx: V0_man[fixed_idx], handle_idx: target}
    V_new = arap_solve_cot(V0_man, F_man, cons, iters=5)  # switched to cot
    V_current_man[:] = V_new
    redraw_man()
```

Summary of what you get
- More stable and shape-preserving deformations on irregular meshes.
- Same interface: constraints and iterations remain the same.
- You can easily toggle between uniform and cotangent approaches by calling arap_solve vs. arap_solve_cot.

### Question:
Can you Implement pre-factorization of the system matrix for speed with moving handles.

### Answer:
Below are focused additions to your notebook that pre-factor the cotangent system matrix once and reuse the factorization for every drag update. This makes interactive dragging much faster because only the RHS changes when the handle moves.

What we’ll do
- Build the cotangent Laplacian L once (from rest V0 and faces F).
- Impose hard constraints by row-pinning into a modified matrix Lc (we must know the fixed/handle indices to pin).
- LU-factor Lc once.
- During dragging, rebuild only the RHS Bc from the current handle target and solve via the prefactorization.

1) Add a small utility to build the constrained system matrix Lc and an LU factorization function

```python
# File: D:/ProgrammingPython/ZGL Projects/deform2d/arap.ipynb
#%%
import scipy.linalg as sla

def build_constrained_system(L, constraints):
    """
    Returns a copy Lc of L with hard constraints imposed by row pinning:
    For each constrained vertex i, zero the row and set Lc[i,i] = 1.
    Note: We do not modify columns; RHS will be set to target positions.
    """
    Lc = L.copy()
    for idx in constraints.keys():
        Lc[idx, :] = 0.0
        Lc[idx, idx] = 1.0
    return Lc

def lu_factorize(A):
    """
    LU factorization wrapper (SciPy) for reuse across solves.
    """
    return sla.lu_factor(A, check_finite=False)

def lu_solve_factor(lu_piv, b):
    """
    Solve A x = b given lu_piv from lu_factorize(A).
    """
    return sla.lu_solve(lu_piv, b, check_finite=False)
```

2) Prefactorized cotangent ARAP global step
- This variant expects:
  - A fixed factorization of the constrained L (Lc) for the current set of constrained indices.
  - On each iteration, we recompute rotations and the cotangent RHS B from rest geometry and current V.
  - We then set Bc at constrained rows to the target positions and solve using the LU factors.

```python
# File: D:/ProgrammingPython/ZGL Projects/deform2d/arap.ipynb
#%%
def global_step_cot_prefact(V, V0, F, R_list, constraints, Lc_lu):
    """
    Use pre-factorized constrained cotangent Laplacian:
    - L is fixed (from V0 and F)
    - Constraints indices are fixed; only RHS changes
    """
    # Build cotangent RHS from rotations
    B = build_rhs_cot(V0, F, R_list)
    # Apply constraints to RHS only (rows were pinned in Lc already)
    Bc = B.copy()
    for idx, pos in constraints.items():
        Bc[idx, :] = pos
    # Solve two independent systems with LU factors
    X = lu_solve_factor(Lc_lu, Bc[:, 0])
    Y = lu_solve_factor(Lc_lu, Bc[:, 1])
    return np.column_stack((X, Y))
```

3) A solver that reuses the prefactorization across ARAP iterations
- The constraints set is assumed fixed during the solve (index set doesn’t change inside this routine).

```python
# File: D:/ProgrammingPython/ZGL Projects/deform2d/arap.ipynb
#%%
def arap_solve_cot_prefact(V0, F, constraints, Lc_lu, iters=5):
    """
    ARAP loop using cotangent weights with pre-factorized constrained system matrix.
    The LU factor Lc_lu corresponds to build_constrained_system(cotangent_laplacian(V0,F), constraints).
    """
    V = V0.copy()
    for _ in range(iters):
        R_list = local_step(V, V0, F, edge_weight='cot')
        V = global_step_cot_prefact(V, V0, F, R_list, constraints, Lc_lu)
    return V
```

4) Precompute and factor Lc for man.obj once, then use it in the interactive dragging
- Build L from V0_man and F_man once.
- Construct constraints dict with fixed_idx and handle_idx (for interactive we’ll change only the handle’s target position, not indices).
- Build Lc and factor it once.
- Switch on_move_man to reuse Lc_lu.

Insert these where you set up the interactive demo for man.obj (after you define left_idx, right_idx, etc.):

```python
# File: D:/ProgrammingPython/ZGL Projects/deform2d/arap.ipynb
#%%
# Precompute cotangent Laplacian at rest
L_cot_man = cotangent_laplacian(V0_man, F_man)

# For interactive use, the constrained index set must be fixed to reuse Lc.
# We'll constrain: fixed_idx (left-most) and handle_idx (right-most).
fixed_idx = left_idx
handle_idx = right_idx

# We won't set the handle target value here (it changes during drag), just the indices.
constraints_indices = {fixed_idx: V0_man[fixed_idx], handle_idx: V0_man[handle_idx]}

# Build constrained system and factorize once
Lc_man = build_constrained_system(L_cot_man, constraints_indices)
Lc_lu_man = lu_factorize(Lc_man)
```

Now update your interactive callbacks to use the prefactorized solver:

Replace your existing on_move_man with this version that uses arap_solve_cot_prefact:

```python
# File: D:/ProgrammingPython/ZGL Projects/deform2d/arap.ipynb
#%%
def on_move_man(event):
    global target
    if not dragging['active'] or event.inaxes != ax:
        return
    target = np.array([event.xdata, event.ydata])
    # Only target changes; indices fixed, so we can reuse Lc_lu_man
    cons = {fixed_idx: V0_man[fixed_idx], handle_idx: target}
    V_new = arap_solve_cot_prefact(V0_man, F_man, cons, Lc_lu_man, iters=5)
    V_current_man[:] = V_new
    redraw_man()
```

5) Optional: also use prefactorization for the non-interactive run on man.obj
If you’d like the “static” solve to use the same prefactorization:

```python
# File: D:/ProgrammingPython/ZGL Projects/deform2d/arap.ipynb
#%%
constraints_man_cot = {
    left_idx: V0_man[left_idx],
    right_idx: V0_man[right_idx] + drag_delta
}
# Reuse indices but change the right handle target; factor still valid:
V_man_deformed_cot_fast = arap_solve_cot_prefact(V0_man, F_man, constraints_man_cot, Lc_lu_man, iters=10)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
plot_mesh(V0_man, F_man, ax=ax[0], color='gray', title='Rest (man.obj)')
ax[0].scatter([V0_man[left_idx, 0]], [V0_man[left_idx, 1]], color='green', s=40, label='fixed')
ax[0].scatter([V0_man[right_idx, 0]], [V0_man[right_idx, 1]], color='orange', s=40, label='dragged (rest)')
ax[0].legend()

plot_mesh(V_man_deformed_cot_fast, F_man, ax=ax[1], color='C2', title='ARAP (cot, prefactored) — 10 iters')
ax[1].scatter([constraints_man_cot[right_idx][0]], [constraints_man_cot[right_idx][1]], color='red', s=60,
              label='handle target')
ax[1].legend()
plt.show()
```

Important notes
- Prefactorization requires a fixed set of constrained indices. During a single drag session, we keep the same fixed_idx and handle_idx; only the handle target position changes. If you add/remove handles or change which vertex indices are constrained, rebuild Lc and refactor.
- Using the rest-geometry cotangent Laplacian is standard in ARAP. If you change V0 (rest), rebuild L.
- This mirrors your PyQt app’s approach in src/rigid_mesh_deformer.py, where LU factors are cached and only RHS changes as handles move.

### Question:
Can you Extend to multiple handles and constraints.

### Answer:
Below are minimal, targeted additions to support any number of handles/constraints with the prefactored cotangent solver. The key is: the constrained index set must be fixed while reusing a factorization. You can still update the target positions at those indices every drag.

1) Helper to build constraints dict from a list of constrained indices
Use this to construct constraints quickly for any number of handles.

```python
# File: D:/ProgrammingPython/ZGL Projects/deform2d/arap.ipynb
#%%
def constraints_from_indices(indices, V_targets):
    """
    Build a constraints dict from:
      indices: iterable of vertex indices to constrain
      V_targets: (n,2) array of target positions, where only rows at 'indices' are used.
    Returns: dict {idx: V_targets[idx]}
    """
    return {int(i): np.asarray(V_targets[int(i)], dtype=np.float64) for i in indices}
```

2) Build a prefactorization for an arbitrary fixed set of constrained indices
You can now choose any set of indices to constrain (e.g., multiple fixed pins and multiple draggable handles). Factor once, reuse for RHS updates.

```python
# File: D:/ProgrammingPython/ZGL Projects/deform2d/arap.ipynb
#%%
# Example: choose multiple handles on man.obj
# - Fixed/pinned: left-most vertex + one more (e.g., lowest y)
# - Dragged handles: right-most vertex + top-most vertex
left_idx  = int(np.argmin(V0_man[:, 0]))
low_idx   = int(np.argmin(V0_man[:, 1]))
right_idx = int(np.argmax(V0_man[:, 0]))
top_idx   = int(np.argmax(V0_man[:, 1]))

fixed_indices   = [left_idx, low_idx]         # always pinned to their rest positions (or custom targets)
draggable_indices = [right_idx, top_idx]      # will move during demo

all_constrained_indices = fixed_indices + draggable_indices

# Precompute Laplacian and prefactor with the index set fixed
L_cot_man_multi = cotangent_laplacian(V0_man, F_man)
# Build a dummy constraints dict just to pin rows for factorization (values are placeholders)
dummy_constraints_multi = {i: V0_man[i] for i in all_constrained_indices}
Lc_man_multi = build_constrained_system(L_cot_man_multi, dummy_constraints_multi)
Lc_lu_man_multi = lu_factorize(Lc_man_multi)
```

3) Static solve with multiple handles
Provide specific target positions for any subset of the constrained indices (others can remain at rest or also be moved). Then solve using the same prefactorization.

```python
# File: D:/ProgrammingPython/ZGL Projects/deform2d/arap.ipynb
#%%
# Example targets:
# - Fix 'fixed_indices' at rest positions
# - Move 'right_idx' slightly right; move 'top_idx' slightly up
span_x = (V0_man[:, 0].max() - V0_man[:, 0].min() + 1e-6)
span_y = (V0_man[:, 1].max() - V0_man[:, 1].min() + 1e-6)

targets_multi = V0_man.copy()
targets_multi[right_idx] = V0_man[right_idx] + np.array([0.1 * span_x, 0.0])
targets_multi[top_idx]   = V0_man[top_idx]   + np.array([0.0, 0.1 * span_y])

constraints_multi = constraints_from_indices(all_constrained_indices, targets_multi)

V_man_deformed_multi = arap_solve_cot_prefact(V0_man, F_man, constraints_multi, Lc_lu_man_multi, iters=10)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
plot_mesh(V0_man, F_man, ax=ax[0], color='gray', title='Rest (man.obj)')
ax[0].scatter(V0_man[fixed_indices, 0], V0_man[fixed_indices, 1], color='green', s=40, label='fixed')
ax[0].scatter(V0_man[draggable_indices, 0], V0_man[draggable_indices, 1], color='orange', s=40, label='dragged (rest)')
ax[0].legend()

plot_mesh(V_man_deformed_multi, F_man, ax=ax[1], color='C3', title='ARAP (cot, multiple handles) — 10 iters')
ax[1].scatter(targets_multi[draggable_indices, 0], targets_multi[draggable_indices, 1],
              color='red', s=60, label='handle targets')
ax[1].legend()
plt.show()
```

4) Interactive multi-handle dragging (drag any of the draggable handles)
This example lets you drag either of the chosen draggable handles (right_idx, top_idx). It keeps the fixed set of constrained indices identical, so the same LU factorization is reused. Only the target positions in constraints_multi change per mouse move.

```python
# File: D:/ProgrammingPython/ZGL Projects/deform2d/arap.ipynb
#%%
%matplotlib notebook

V_current_man = V0_man.copy()
current_targets = V0_man.copy()  # start at rest
# Initialize current targets for draggable handles
current_targets[right_idx] = V0_man[right_idx] + np.array([0.1 * span_x, 0.0])
current_targets[top_idx]   = V0_man[top_idx]   + np.array([0.0, 0.1 * span_y])

fig, ax = plt.subplots()
plot_mesh(V0_man, F_man, ax=ax, color='gray', title='Drag the orange handle targets (multi)')
plot_mesh(V_current_man, F_man, ax=ax, color='C0')
fixed_pts = ax.scatter(V0_man[fixed_indices, 0], V0_man[fixed_indices, 1], color='green', s=40, label='fixed')
target_pts = ax.scatter(current_targets[draggable_indices, 0], current_targets[draggable_indices, 1],
                        color='orange', s=60, label='targets')
dragging = {'active': False, 'which': None}

def redraw_man_multi():
    ax.cla()
    plot_mesh(V0_man, F_man, ax=ax, color='gray')
    plot_mesh(V_current_man, F_man, ax=ax, color='C0')
    ax.scatter(V0_man[fixed_indices, 0], V0_man[fixed_indices, 1], color='green', s=40)
    ax.scatter(current_targets[draggable_indices, 0], current_targets[draggable_indices, 1], color='orange', s=60)
    ax.set_title('Drag the orange handle targets (multi)')
    fig.canvas.draw_idle()

def nearest_draggable_handle(p):
    # Returns index in draggable_indices array (0..len-1) of closest handle target, or None
    pts = current_targets[draggable_indices]
    d = np.linalg.norm(pts - p, axis=1)
    k = np.argmin(d)
    if d[k] < 0.05:  # pick radius
        return int(draggable_indices[k])
    return None

def on_press_man_multi(event):
    if event.inaxes != ax: return
    p = np.array([event.xdata, event.ydata])
    which = nearest_draggable_handle(p)
    if which is not None:
        dragging['active'] = True
        dragging['which'] = which

def on_release_man_multi(event):
    dragging['active'] = False
    dragging['which'] = None

def on_move_man_multi(event):
    if not dragging['active'] or event.inaxes != ax:
        return
    p = np.array([event.xdata, event.ydata])
    h = dragging['which']
    # Update just that handle target
    current_targets[h] = p

    # Build constraints for all constrained indices using current targets
    cons = constraints_from_indices(all_constrained_indices, current_targets)
    V_new = arap_solve_cot_prefact(V0_man, F_man, cons, Lc_lu_man_multi, iters=5)
    V_current_man[:] = V_new
    redraw_man_multi()

cid1 = fig.canvas.mpl_connect('button_press_event', on_press_man_multi)
cid2 = fig.canvas.mpl_connect('button_release_event', on_release_man_multi)
cid3 = fig.canvas.mpl_connect('motion_notify_event', on_move_man_multi)
redraw_man_multi()
```

Notes and tips
- The LU factorization is valid as long as the set of constrained indices stays the same. If you add/remove handles or switch which vertices are constrained, rebuild Lc and refactor using the same utilities shown above.
- Targets for any constrained index can change freely per frame; only the RHS is updated.
- You can mix “fixed” pins and “dragged” handles; both are simply entries in the constraints dict.

This pattern mirrors your app’s approach: once handles are selected (index set fixed), reuse the LU factors during dragging, recomputing only rotations and the RHS.

### Question:
Please compare results visually with your PyQt app by exporting intermediate states.

### Answer:
Below are small, self-contained additions to your notebook to export intermediate ARAP states so you can compare visually in your PyQt app (or any viewer). You can export:
- Per-iteration deformed vertex positions during a solve.
- Snapshots during interactive dragging.
- Simple OBJ or NumPy formats.

Add these helpers near your ARAP functions

```python
# File: D:/ProgrammingPython/ZGL Projects/deform2d/arap.ipynb
#%%
import os

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_vertices_npy(path, V):
    """
    Save 2D vertex array as .npy; shape (n,2)
    """
    ensure_dir(os.path.dirname(path))
    np.save(path, V.astype(np.float32))

def save_mesh_obj(path, V, F):
    """
    Save 2D mesh as .obj (z=0). Faces are 1-based.
    """
    ensure_dir(os.path.dirname(path))
    with open(path, 'w') as f:
        for x, y in V:
            f.write(f"v {x:.6f} {y:.6f} 0.0\n")
        for a, b, c in F:
            f.write(f"f {a+1} {b+1} {c+1}\n")
```

Export per-iteration states for batch solves

```python
# File: D:/ProgrammingPython/ZGL Projects/deform2d/arap.ipynb
#%%
def arap_solve_cot_prefact_export(V0, F, constraints, Lc_lu, out_dir=None, iters=5, fmt='npy'):
    """
    Same as arap_solve_cot_prefact but optionally saves V after each iteration.
    fmt: 'npy' or 'obj'
    """
    V = V0.copy()
    if out_dir is not None:
        ensure_dir(out_dir)
        # Save the rest state for reference
        if fmt == 'npy':
            save_vertices_npy(os.path.join(out_dir, "iter_00_rest.npy"), V0)
        else:
            save_mesh_obj(os.path.join(out_dir, "iter_00_rest.obj"), V0, F)

    for it in range(1, iters+1):
        R_list = local_step(V, V0, F, edge_weight='cot')
        V = global_step_cot_prefact(V, V0, F, R_list, constraints, Lc_lu)

        if out_dir is not None:
            if fmt == 'npy':
                save_vertices_npy(os.path.join(out_dir, f"iter_{it:02d}.npy"), V)
            else:
                save_mesh_obj(os.path.join(out_dir, f"iter_{it:02d}.obj"), V, F)
    return V
```

Example: export a 10-iteration sequence on man.obj with multiple handles

```python
# File: D:/ProgrammingPython/ZGL Projects/deform2d/arap.ipynb
#%%
export_dir = "exports/man_multi_prefact"
V_export = arap_solve_cot_prefact_export(
    V0_man, F_man,
    constraints_multi,           # built earlier with constraints_from_indices
    Lc_lu_man_multi,             # prefactor for fixed constrained index set
    out_dir=export_dir,
    iters=10,
    fmt='obj'                    # or 'npy'
)
print("Exported per-iteration OBJ files to:", export_dir)
```

Export snapshots during interactive dragging
- Hook into your existing interactive multi-handle callback and save a snapshot every N frames or on mouse release.

Option A: Save on mouse release

```python
# File: D:/ProgrammingPython/ZGL Projects/deform2d/arap.ipynb
#%%
snap_dir = "exports/man_multi_interactive"
snap_counter = {'k': 0}

def save_snapshot(V, name_prefix="snap"):
    k = snap_counter['k']
    save_mesh_obj(os.path.join(snap_dir, f"{name_prefix}_{k:04d}.obj"), V, F_man)
    snap_counter['k'] += 1

def on_release_man_multi(event):
    dragging['active'] = False
    dragging['which'] = None
    # Save a snapshot of current deformed mesh
    ensure_dir(snap_dir)
    save_snapshot(V_current_man, "release")
```

Option B: Save every N moves (lightweight)

```python
# File: D:/ProgrammingPython/ZGL Projects/deform2d/arap.ipynb
#%%
save_every = 10
move_counter = {'c': 0}

def on_move_man_multi(event):
    if not dragging['active'] or event.inaxes != ax:
        return
    p = np.array([event.xdata, event.ydata])
    h = dragging['which']
    current_targets[h] = p
    cons = constraints_from_indices(all_constrained_indices, current_targets)
    V_new = arap_solve_cot_prefact(V0_man, F_man, cons, Lc_lu_man_multi, iters=5)
    V_current_man[:] = V_new
    redraw_man_multi()

    move_counter['c'] += 1
    if move_counter['c'] % save_every == 0:
        ensure_dir(snap_dir)
        save_snapshot(V_current_man, "move")
```

How to compare in your PyQt app
- If you choose OBJ:
  - Your app already supports loading OBJ via pressing ‘f’. You can step through exported iter_XX.obj or snap_XXXX.obj files and visually compare deformations with the in-notebook result.
- If you choose NPY:
  - Add a small loader in your app to set mesh vertices from a .npy file, or convert NPYs to OBJ offline with save_mesh_obj.

Notes
- Prefactorization remains valid as long as the constrained index set doesn’t change. If you add/remove handles in the notebook, rebuild Lc and refactor before exporting further frames.
- For large meshes, OBJ writing can be slower; NPY is faster and lossless. You can convert selected frames to OBJ for spot checks.

This approach will let you line up the notebook’s frames with what you see in your PyQt app and verify the deformation behavior under identical constraints.