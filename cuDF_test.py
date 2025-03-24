import cudf
print(cudf.__version__)
df = cudf.DataFrame({'test': [1]})
print(df)