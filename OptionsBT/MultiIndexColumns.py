#How to create multiindex columns
tl=[('SPOT', x) for x in bn_df.columns[0:9]]
t2 = [('CALL', x) for x in bn_df.columns[9:13]]
t3 = [('PUT', x) for x in bn_df.columns[13:]]
tup = t1 + t2 + t3
mi = pd.MultiIndex.from_tuples(tup)
bn_df.columns = mi
bn_df.to_excel('bn_df.xlsx')