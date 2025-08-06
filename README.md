# boxtrot

A wrapper around prtree for box intersection querries.


```python
from boxplot import BoxIntersector

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

bi.query(query_boxes)
```