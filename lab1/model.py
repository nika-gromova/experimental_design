import numpy.random as nr
import math
import matplotlib.pyplot as plt


def intensity_to_param(intensity):
    return 10 / intensity


def normal_distribution(params):
    time = nr.normal(params[0], params[1])
    return abs(time)


class Generator:
    def __init__(self, law, params):
        self.func = law
        self.params = params

    def generate(self):
        return self.func(self.params)


class Processor:
    def __init__(self, law, params):
        self.func = law
        self.params = params
        self.busy = False

    def generate(self):
        return self.func(self.params)


class Model:
    def __init__(self, duration=20, generators_params=[[3, 0.5]], processors_params=[[2, 0.5]]):
        self.start_time = 0
        self.end_time = duration

        self.generators_count = len(generators_params)
        self.generators = []
        for i in range(self.generators_count):
            self.generators.append(Generator(normal_distribution, generators_params[i]))
        self.processors_count = len(processors_params)
        self.processors = []
        for i in range(self.processors_count):
            self.processors.append(Processor(normal_distribution, processors_params[i]))
        self.generated = 0
        self.queue_length = 0
        self.queue_len_max = 0
        self.avg_waiting_time = 0
        self.processed = 0
        self.started_processing = 0
        self.events = []

    def reset(self):
        self.generated = 0
        self.queue_length = 0
        self.queue_len_max = 0
        self.avg_waiting_time = 0
        self.processed = 0
        self.started_processing = 0
        self.events = []
        for processor in self.processors:
            processor.busy = False

    def check_len_max(self):
        if self.queue_length > self.queue_len_max:
            self.queue_len_max = self.queue_length

    # event: [time of event, type - 'p' or 'g', index in generators or processors lists, waiting time]

    def add_event(self, event):
        i = 0
        while i < len(self.events) and event[0] >= self.events[i][0]:
            i += 1
        self.events.insert(i, event)

    def modelling(self):
        for i in range(self.generators_count):
            self.add_event([self.start_time, 'g', i, 0])
            self.queue_length += 1
            self.generated += 1
            self.check_len_max()
        cur_time = self.start_time
        while cur_time < self.end_time:
            event = self.events.pop(0)
            cur_time = event[0]
            if event[1] == 'g':
                self.start_processing(event)
            else:
                self.finish_operate(event)
        self.avg_waiting_time /= self.started_processing
        return self.avg_waiting_time, self.processed, self.generated

    def start_processing(self, event):
        i = 0

        # find free processor
        while i < self.processors_count and self.processors[i].busy:
            i += 1

        # if found free one:
        if i != self.processors_count:
            self.queue_length -= 1
            self.processors[i].busy = True
            self.add_event([event[0] + self.processors[i].generate(), 'p', i, 0])
            self.started_processing += 1
            self.avg_waiting_time += event[3]

        # if there is no free processors:
        else:
            j = 0
            while j < len(self.events) and self.events[j][1] != 'p':
                j += 1
            self.add_event([self.events[j][0], 'g', event[2], event[3] + self.events[j][0] - event[0]])

        # generating new events
        if event[3] == 0:
            self.add_event([event[0] + self.generators[event[2]].generate(), 'g', event[2], 0])
            self.queue_length += 1
            self.generated += 1
            self.check_len_max()

    def finish_operate(self, event):
        self.processors[event[2]].busy = False
        self.processed += 1


def get_avg_model(model, times):
    avg_waiting_time = 0
    for i in range(times):
        result_avg_time, result_processed, tmp = model.modelling()
        avg_waiting_time += result_avg_time
        model.reset()
    return avg_waiting_time / times


def get_plot(lambda1, d):
    ro_array = []
    wait_time = []
    ro = 0.05
    generators_params = [[10 / lambda1, d]]

    while ro <= 1:
        ro_array.append(ro)
        lambda2 = lambda1 / ro
        processors_params = [[10 / lambda2, d]]
        model = Model(50, generators_params, processors_params)
        avg_time = get_avg_model(model, 50)
        wait_time.append(avg_time)
        ro += 0.05

    plt.plot(ro_array, wait_time)
    plt.xlabel("загрузка")
    plt.ylabel("время пребывания")
    plt.show()
