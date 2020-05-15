from scipy.spatial import distance
p1 = (5.4, 3.9, 1.7, 0.4)
p2 = (4.4, 2.9, 1.4, 0.2)
d = distance.euclidean(p1, p2)
print("Euclidean distance: ",d * d)
print((1-0.5) * d * d) 
print((1+0.5) * d * d) 
p1 = (0.67922914, -1.09633771, 0.76805051)
p2 = (0.53813813, -0.84238446, 0.62076123)
d = distance.euclidean(p1, p2)
print("Euclidean distance: ",d * d)

