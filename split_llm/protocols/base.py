class Protocol:
    def prepare(self):
        raise NotImplementedError()
    
    def offline_execute(self):
        raise NotImplementedError()
    
    def online_execute(self):
        raise NotImplementedError()
    
    def clear_io(self):
        raise NotImplementedError()