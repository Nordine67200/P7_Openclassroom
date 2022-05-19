import os

def getDirectoryPath(directoryName: str):
    for root, dirs, files in os.walk("/"):
        for dir in dirs:
            print(dir)
            if dir == directoryName:
                return os.path.join(os.getcwd(), dir)