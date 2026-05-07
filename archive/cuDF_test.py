import cudf
import cupy as cp

# Test cuDF
df = cudf.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
print(df)

# Test CuPy
x = cp.array([1, 2, 3])
print(x)