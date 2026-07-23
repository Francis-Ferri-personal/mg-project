import json

class JSONLREADER:
    def __init__(self, path):
        self.path = path

    @staticmethod
    def __getindices(path) -> list[int]:
        indices = []
        with open(path,'r') as f:
            seek_pos = 0
            while True:
                f.seek(seek_pos)
                
                c=f.readline()
                if not len(c):
                    break

                indices.append(seek_pos)
                seek_pos = f.tell()
        return indices
    
    def get_indices(self):
        self.indices = getattr(self,'indices',self.__getindices(self.path))
        return self
    
    @staticmethod
    def read_from(path:str, index:int, indices = None) -> dict:
        if indices is None:
            indices = JSONLREADER.__getindices(path)
        
        with open(path,'rb') as f:
            f.seek(indices[index])
            return json.loads(f.readline())
        
    def read(self, index) -> dict:
        if getattr(self,'indices',None) is None:
            self.indices = self.__getindices(self.path)
        
        return self.read_from(self.path, index, self.indices)
    
    def __getitem__(self, key):
        return self.read(key)