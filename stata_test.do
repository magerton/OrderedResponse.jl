import delimited D:\projects\OrderedResponse.jl\data\testdat.csv

oprobit y x1 x2
mat list e(b)
mat list e(V), nohalf

ologit y x1 x2
mat list e(b)
mat list e(V), nohalf
