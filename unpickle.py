import pickle
import sys

def unpickle(f):
    with open(f, "rb") as picklefile:
        data = pickle.load(picklefile)
        return data

def main():
    if len(sys.argv) !=2:
        print("Usage: python unpickle.py <picklefile>")
    file = sys.argv[1]
    data = unpickle(file)
    print(data)

if __name__ == '__main__':
    main()
