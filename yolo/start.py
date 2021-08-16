from multiprocessing import Process


def  clean():
    exec(open("./clean.py").read())


def api():
    exec(open("./api.py").read())


def yolo():
    exec(open("./yolo.py").read())


if __name__ == "__main__":
    clean()

    Process(target=yolo()).start()
    Process(target=api()).start()