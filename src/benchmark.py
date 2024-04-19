import json, os
from timeit import default_timer as timer


class Benchmark:
    """
    It's a class that can be used to benchmark the performance of a function
    """

    def __init__(self):
        self.runtime_data = {}
        self.verbose = True
        counter = 0
        for file in os.listdir('data/results/mind2web/runtime/'):
            if file.endswith('.json'):
                counter += 1
        self.file = 'runtime-{}.json'.format(counter)
        
    def set_verbose(self):
        self.verbose = not self.verbose

    def mark(self, message=''):
        """
        The first call sets self.start to the current time. Subsequent calls cause return (and print if verbose is set to True) the runtime.
        
        :param message: The message to print
        :return: The time it took to run the code
        """
        if message not in self.runtime_data:
            self.runtime_data[message] = {'starts': [timer()], 'ends': [], 'times': []}
        else:
            if len(self.runtime_data[message]['ends']) < len(self.runtime_data[message]['starts']):
                self.runtime_data[message]['ends'].append(timer())
                self.time = self.runtime_data[message]['ends'][-1] - self.runtime_data[message]['starts'][-1]
                self.runtime_data[message]['times'].append(self.time)
                if self.verbose:
                    print('{message:{fill}{align}{width}}-{time}'
                        .format(message=message, fill='-', align='<', width=30, time=self.time))
            else:
                self.runtime_data[message]['starts'].append(timer())
            return self.time
    
    def write_to_file(self):
        with open('data/results/mind2web/runtime/{}'.format(self.file), 'w') as outfile:
            json.dump(self.runtime_data, outfile, indent=4)


BM = Benchmark()