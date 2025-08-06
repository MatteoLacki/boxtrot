from __future__ import annotations

import duckdb
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
        stored_boxes_tree_maker = getattr(python_prtree, f"PRTree{self.dim}D")
        self.stored_boxes = stored_boxes
        self.stored_boxes_tree = stored_boxes_tree_maker(
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
