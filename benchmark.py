from timeit import default_timer as timer


class Benchmark:
    """
    It's a class that can be used to benchmark the performance of a function
    """

    def __init__(self):
        self.starts = {}
        self.ends = {}
        self.verbose = True
        
    def set_verbose(self):
        self.verbose = not self.verbose

    def mark(self, message=''):
        """
        The first call sets self.start to the current time. Subsequent calls cause return (and print if verbose is set to True) the runtime.
        
        :param message: The message to print
        :return: The time it took to run the code
        """
        if message not in self.starts:
            self.starts[message] = timer()
        else:
            self.ends[message] = timer()
            self.time = self.ends[message] - self.starts[message]
            if self.verbose:
                print('{message:{fill}{align}{width}}-{time}'
                      .format(message=message, fill='-', align='<', width=30, time=self.time))
            del self.starts[message]
            del self.ends[message]
            return self.time


BM = Benchmark()