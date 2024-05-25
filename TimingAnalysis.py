import time

class TimeAnalysis():
    def __init__(self):
        self.__time_variables = {}
        self.__disabled = False

    def begin(self, interval_name):
        if interval_name in self.__time_variables:
            raise NameError("This interval name already started. Please begin with a different interval name.")
        else:
            self.__time_variables[interval_name] = [time.time(), None]
    
    def end(self, interval_name, existing_time=None):
        if interval_name not in self.__time_variables:
            raise NameError(f'{interval_name} has not been started. Please start the interval first')
        elif self.__time_variables[interval_name][1] is not None:
            self.__time_variables[interval_name][1].append(existing_time if existing_time is not None else time.time())
        else:
            self.__time_variables[interval_name][1] = [existing_time if existing_time is not None else time.time()]

    def disable(self):
        self.__disabled = True
    
    def enable(self):
        self.__disabled = False

    def print_analysis(self):
        if not self.__disabled:
            print("================== Timing Analysis ==================")

            for k, v in list(self.__time_variables.items()):
                if v[1] is None:
                    print(f'{k} has not ended. It has now been running for {time.time() - v[0]}')
                elif len(v[1]) == 1:
                    print(f'{k} duration:', v[1][0] - v[0])
                else:
                    for i in range(len(v[1])):
                        print(f'The {i+1}th {k} process duration:', v[1][i] - v[0])

            print("=====================================================")
