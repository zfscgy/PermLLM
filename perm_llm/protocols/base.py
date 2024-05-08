class Protocol:
    def prepare(self):
        raise NotImplementedError()
    
    def offline_execute(self):
        raise NotImplementedError()
    
    def online_execute(self):
        raise NotImplementedError()
    
    def clear_io(self):
        raise NotImplementedError()
    
    def reset(self):
        # To clear all offline caches
        raise NotImplementedError()