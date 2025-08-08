from __future__ import annotations

import duckdb
import numba
import python_prtree

import numpy as np
import pandas as pd

from warnings import warn


def get_boxes_df_dim(boxes_df: pd.DataFrame) -> int:
    """Get the dimesion resulting from the box df."""
    _2D = boxes_df.shape[1]
    div, mod = np.divmod(_2D, 2)
    assert mod == 0
    return int(div)


def try_getting_table_top_row(
    con: duckdb.duckdb.DuckDBPyConnection, sql: str
) -> pd.DataFrame:
    """Try returning only the first row in an sql query result."""
    try:
        header = con.execute(sql + " LIMIT 1").df()
    except Exception as e:
        warn("Could not have appeneded ` LIMIT 1` to the `sql` query.")
        header = con.execute(sql).df()
    assert len(header) >= 1
    try:
        return header.set_index("IDX")
    except KeyError as e:
        print("Expecting IDX column.")
        raise e


class BoxIntersector:
    """
    A wrapper around `python_prtree.PRTree{self.dim}D`.

    The results are given in terms of stored_boxes indices (that is why we use their indices).
    """

    def __init__(
        self,
        stored_boxes: pd.DataFrame,
        _query_boxes_sql: str | None = None,
    ):
        self.dim = get_boxes_df_dim(stored_boxes)
        self.stored_boxes = stored_boxes
        self.stored_boxes_tree = getattr(python_prtree, f"PRTree{self.dim}D")(
            stored_boxes.index.to_numpy(),
            stored_boxes.to_numpy(),
        )
        self._query_boxes_sql = _query_boxes_sql

    @classmethod
    def from_sqls(
        cls,
        stored_boxes_sql: str,
        query_boxes_sql: str,
        duckcon: duckdb.duckdb.DuckDBPyConnection,
    ) -> BoxIntersector:
        """Instantiate using sql queries using duckdb."""
        query_boxes_header = try_getting_table_top_row(duckcon, query_boxes_sql)
        stored_boxes_header = try_getting_table_top_row(duckcon, stored_boxes_sql)

        assert all(
            query_boxes_header.columns == stored_boxes_header.columns
        ), f"Mismatch of columns: `stored_boxes_sql` and `query_boxes_sql` should results in the same column names and their order.\nquery_boxes_header: {list(query_boxes_header.columns)}\nstored_boxes_header: {list(stored_boxes_header.columns)}."

        dim = get_boxes_df_dim(stored_boxes_header)

        # checking boxes match pattern [ <col_0>_min, <col_1>_min, .. <col_0>_max, <col_1>_max, .. ]
        for i, (col_min, col_max) in enumerate(
            zip(
                stored_boxes_header.columns[0:dim],
                stored_boxes_header.columns[dim : 2 * dim],
            )
        ):
            _dim0, _min = col_min.rsplit("_", 1)
            _dim1, _max = col_max.rsplit("_", 1)
            assert (
                _dim0 == _dim1
            ), f"Expecting boxes to be specified as [ <col_0>_min, <col_1>_min, .. <col_0>_max, <col_1>_max, .. ]. Instead, got col_{i} taking 2 values: `{_dim0}` and `{_dim1}`. Redifine your SQLs."
            assert (
                _min == "min"
            ), f"Expecting boxes to be specified as [ <col_0>_min, <col_1>_min, .. <col_0>_max, <col_1>_max, .. ]. But instead of `min` got {_min}."
            assert (
                _max == "max"
            ), f"Expecting boxes to be specified as [ <col_0>_min, <col_1>_min, .. <col_0>_max, <col_1>_max, .. ]. But instead of `max` got {_max}."

        stored_boxes = duckcon.query(stored_boxes_sql).df().set_index("IDX")

        return cls(stored_boxes, _query_boxes_sql=query_boxes_sql)

    def query(self, query_boxes: pd.DataFrame) -> list[list[int]]:
        """Query the stored boxes with new boxes.

        Arguments:
            query_boxes (pd.DataFrame): A data frame with boxes to match.

        Returns:
            list[list]: For each row of the passed in query_boxes, a list with indices of the matching stored boxes.
        """
        for submitted_col, stored_col in zip(
            query_boxes.columns, self.stored_boxes.columns
        ):
            assert (
                submitted_col == stored_col
            ), f"Your query boxes have different columns or their order is different than those stored in the tree:\nquery_boxes = {list(query_boxes.columns)}\nstored_boxes ={list(self.stored_boxes.columns)}"
        return self.stored_boxes_tree.batch_query(query_boxes.to_numpy())

    def iterated_query(
        self,
        duckcon: duckdb.duckdb.DuckDBPyConnection,
        chunk_size: int = 1_000_000,
        query_boxes_sql: str | None = None,
    ):
        query_boxes_sql = (
            query_boxes_sql if not query_boxes_sql is None else self._query_boxes_sql
        )
        assert (
            query_boxes_sql is not None
        ), "Either pass in `query_boxes_sql` directly to `iterated_query` or instantiate the constructor with `_query_boxes_sql`."

        column_names = ["IDX", *self.stored_boxes.columns]
        duckcon.execute(query_boxes_sql)
        while True:
            chunk = duckcon.fetchmany(chunk_size)
            if chunk is None or len(chunk) == 0:
                break
            query_boxes = pd.DataFrame(
                chunk, copy=False, columns=column_names
            ).set_index("IDX")
            yield query_boxes, self.query(query_boxes)

            # chunk = duckcon.fetch_df_chunk()
            # if chunk is None or len(chunk) == 0:
            #     break
            # chunk = chunk.rename(columns=dict(MS1_ClusterID="IDX")).set_index("IDX")
            # yield chunk, self.query(chunk)


def generate_random_boxes(n, space_bounds=(0, 100), min_size=1, max_size=10, seed=None):
    """
    Generate N random axis-aligned 3D boxes in a pandas DataFrame.

    Parameters:
    - n (int): number of boxes
    - space_bounds (tuple): (min_coord, max_coord) for box placement
    - min_size (float): minimum length of each box side
    - max_size (float): maximum length of each box side
    - seed (int or None): random seed for reproducibility

    Returns:
    - pandas.DataFrame with columns: x_min, y_min, z_min, x_max, y_max, z_max
    """
    if seed is not None:
        np.random.seed(seed)

    low, high = space_bounds
    box_data = []

    for _ in range(n):
        # Random origin (min point)
        x_min = np.random.uniform(low, high - max_size)
        y_min = np.random.uniform(low, high - max_size)
        z_min = np.random.uniform(low, high - max_size)

        # Random size
        dx = np.random.uniform(min_size, max_size)
        dy = np.random.uniform(min_size, max_size)
        dz = np.random.uniform(min_size, max_size)

        # Cap max coordinates to space_bounds
        x_max = min(x_min + dx, high)
        y_max = min(y_min + dy, high)
        z_max = min(z_min + dz, high)

        box_data.append(
            {
                "x_min": x_min,
                "y_min": y_min,
                "z_min": z_min,
                "x_max": x_max,
                "y_max": y_max,
                "z_max": z_max,
            }
        )

    return pd.DataFrame(box_data)


@numba.njit
def boxes_intersect(A, B):
    """Return True if A and B intersect in 3D."""
    assert len(A) == len(B)
    assert len(A) % 2 == 0
    D = len(A) // 2
    for i in range(D):
        A_min = A[i]
        A_max = A[i + D]
        B_min = B[i]
        B_max = B[i + D]
        if A_min > B_max or B_min > A_max:
            return False
    return True


@numba.njit
def check_intersections(boxes, query_boxes):
    res = []
    for i in range(len(query_boxes)):
        query_res = []
        for j in range(len(boxes)):
            if boxes_intersect(boxes[j], query_boxes[i]):
                query_res.append(j)
        res.append(query_res)
    return res


# a little bit unoptimal if this does not work correctly.
def test_BoxIntersector():
    boxes = pd.DataFrame(
        {
            "x_min": [0, 5, 10, 3, 8],
            "y_min": [0, 2, 8, 1, 5],
            "z_min": [0, 1, 5, 2, 6],
            "x_max": [2, 7, 12, 6, 10],
            "y_max": [3, 5, 10, 4, 9],
            "z_max": [2, 4, 7, 5, 8],
        }
    )

    bi = BoxIntersector(boxes)

    query_boxes = pd.DataFrame(
        {
            "x_min": [1, 8, 14],
            "y_min": [0, 5, 18],
            "z_min": [0, 6, 20],
            "x_max": [3, 10, 17],
            "y_max": [3, 9, 19],
            "z_max": [2, 8, 21],
        }
    )

    assert bi.query(query_boxes) == check_intersections(
        boxes.to_numpy(), query_boxes.to_numpy()
    )

    for _ in range(1000):
        random_boxes = generate_random_boxes(500)
        random_query_boxes = generate_random_boxes(500)
        random_bi = BoxIntersector(random_boxes)
        prtree_res = random_bi.query(random_query_boxes)
        direct_res = check_intersections(
            random_boxes.to_numpy(), random_query_boxes.to_numpy()
        )
        assert prtree_res == direct_res

    len(prtree_res) == len(direct_res)
    not_the_same = []
    for i in range(len(prtree_res)):
        if prtree_res[i] != direct_res[i]:
            not_the_same.append(i)

    rqb_idx = not_the_same[0]
    prtree_res[rqb_idx]
    direct_res[rqb_idx]

    test_bi = BoxIntersector(pd.DataFrame([random_query_boxes.iloc[rqb_idx]]))
    test_bi.query(pd.DataFrame([random_boxes.iloc[266]]))

    boxes_intersect(random_boxes.iloc[9], random_query_boxes.iloc[0])
    boxes_intersect(random_boxes.iloc[9], random_query_boxes.iloc[0])


def create_box_faces(xmin, ymin, zmin, xmax, ymax, zmax):
    # 8 corner points
    p = [
        [xmin, ymin, zmin],
        [xmax, ymin, zmin],
        [xmax, ymax, zmin],
        [xmin, ymax, zmin],
        [xmin, ymin, zmax],
        [xmax, ymin, zmax],
        [xmax, ymax, zmax],
        [xmin, ymax, zmax],
    ]
    # 6 faces of the box
    return [
        [p[0], p[1], p[2], p[3]],  # bottom
        [p[4], p[5], p[6], p[7]],  # top
        [p[0], p[1], p[5], p[4]],  # front
        [p[2], p[3], p[7], p[6]],  # back
        [p[1], p[2], p[6], p[5]],  # right
        [p[0], p[3], p[7], p[4]],  # left
    ]


def create_box_faces2(xmin, ymin, zmin, xmax, ymax, zmax):
    # 6 faces of the box (each face = 4 corner points)
    return [
        [
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmax, ymax, zmin],
            [xmin, ymax, zmin],
        ],  # bottom
        [
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
            [xmin, ymax, zmax],
        ],  # top
        [
            [xmin, ymin, zmin],
            [xmin, ymin, zmax],
            [xmin, ymax, zmax],
            [xmin, ymax, zmin],
        ],  # left
        [
            [xmax, ymin, zmin],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
            [xmax, ymax, zmin],
        ],  # right
        [
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmax, ymin, zmax],
            [xmin, ymin, zmax],
        ],  # front
        [
            [xmin, ymax, zmin],
            [xmax, ymax, zmin],
            [xmax, ymax, zmax],
            [xmin, ymax, zmax],
        ],  # back
    ]


def plot_boxes(boxes, weights=None, show=True):
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    alphas = None
    if not weights is None:
        # Normalize to [0.1, 1.0] to avoid fully transparent boxes
        alphas = 0.1 + 0.9 * (weights - weights.min()) / (weights.max() - weights.min())
    # Make figure fullscreen
    try:
        manager = plt.get_current_fig_manager()
        # Try full screen depending on backend
        if hasattr(manager, "full_screen_toggle"):
            manager.full_screen_toggle()
        elif hasattr(manager, "window"):
            # TkAgg backend
            manager.window.state("zoomed")
        else:
            # fallback: set size to screen size
            import tkinter as tk

            root = tk.Tk()
            width = root.winfo_screenwidth()
            height = root.winfo_screenheight()
            fig.set_size_inches(width / fig.dpi, height / fig.dpi)
    except Exception as e:
        print(f"Could not set fullscreen: {e}")

    if len(boxes) < 10:
        cmap = cm.get_cmap("tab10", len(boxes))  # up to 10 unique colors
        colors = [cmap(i) for i in range(len(boxes))]

    for i in range(len(boxes)):
        row = boxes.iloc[i].to_numpy()
        faces = create_box_faces(
            row[0],
            row[1],
            row[2],
            row[3],
            row[4],
            row[5],
        )

        kwargs = {"alpha": 0.5}
        if alphas is not None:
            kwargs["alpha"] = alphas[i]

        if len(boxes) < 10:
            kwargs["facecolors"] = colors[i]

        box = Poly3DCollection(
            faces,
            linewidths=0.8,
            edgecolors="k",
            **kwargs,
        )

        ax.add_collection3d(box)
        ax.text(row[0], row[1], row[2], f"{i}", color="black", fontsize=10)

    if len(boxes) < 10:
        handles = [
            plt.Line2D([0], [0], color=colors[i], lw=4) for i in range(len(boxes))
        ]
        labels = [f"Box {i}" for i in range(len(boxes))]
        ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.05, 1))
    else:
        handles = [plt.Line2D([0], [0], lw=4) for i in range(len(boxes))]

    plt.tight_layout()

    if show:
        plt.show()


def plotly_boxes(boxes):
    import plotly.graph_objects as go

    fig = go.Figure()

    use_colors = len(boxes) < 10
    colors = px.colors.qualitative.Plotly if use_colors else None

    for i, row in boxes.iterrows():
        faces = create_box_faces(
            row["x_min"],
            row["y_min"],
            row["z_min"],
            row["x_max"],
            row["y_max"],
            row["z_max"],
        )

        for face in faces:
            x, y, z = zip(*face)
            # Close the loop for polygon
            x += (x[0],)
            y += (y[0],)
            z += (z[0],)

            fig.add_trace(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="lines",
                    line=dict(
                        color=colors[i % len(colors)] if use_colors else "gray", width=4
                    ),
                    showlegend=False,
                )
            )

        # Add box ID as text
        fig.add_trace(
            go.Scatter3d(
                x=[row["x_min"]],
                y=[row["y_min"]],
                z=[row["z_min"]],
                mode="text",
                text=[f"{i}"],
                showlegend=False,
                textposition="top center",
            )
        )

    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        ),
        margin=dict(l=0, r=0, b=0, t=0),
    )
    fig.show()
