from multiprocessing import Process
def f(name):
    print('hello', name)

def g():
    p = Process(target=f, args=('bob',))
    p.start()
    p.join()

