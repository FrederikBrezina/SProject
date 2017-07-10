from multiprocessing import Pool
import numpy as np

def f(f,x,y):

    return [x**2, x*y]

if __name__ == '__main__':
    h = np.zeros((3,2))
    with Pool(5) as p:
       j = p.starmap(f, [[h,0,3], [h,1,3], [h,2,3]])
    for k in range(0, len(j)):
        h[k][:] += j[k][0:len(j[0])]
    print(h)
