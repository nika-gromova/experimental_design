import numpy.random as nr
from scipy.stats import weibull_min


class GaussDistribution:
    def __init__(self, m, s):
        self._m = abs(m)
        self._s = abs(s)

    def generate(self):
        return abs(nr.normal(self._m, self._s))


class RequestGenerator:
    def __init__(self, generator, n=1):
        self._generator = generator
        self._receivers = set()
        self._n = n

    def add_receiver(self, receiver):
        self._receivers.add(receiver)

    def remove_receiver(self, receiver):
        try:
            self._receivers.remove(receiver)
        except KeyError:
            pass

    def next_time(self):
        return self._generator.generate()

    def emit_request(self, time):
        min_rec_qs = float('Inf')

        for rec in self._receivers:
            if rec.queue.current_queue_size < min_rec_qs:
                min_rec_qs = rec.queue.current_queue_size

        for rec in self._receivers:
            if rec.queue.current_queue_size == min_rec_qs:
                rec.receive_request(time, self._n)
                return rec


class Queue:
    def __init__(self):
        self._current_queue_size = 0
        self._max_queue_size = 0
        self._avg_queue_size = 0
        self._avg_recalcs = 0
        self._avg_waiting_time = 0
        self._time_recalcs = 0
        self._arrive_times = []
        self._requests = []

    @property
    def max_queue_size(self):
        return self._max_queue_size

    @property
    def current_queue_size(self):
        return self._current_queue_size

    @property
    def avg_queue_size(self):
        return self._avg_queue_size

    @property
    def avg_waiting_time(self):
        return self._avg_waiting_time

    def add(self, time, n):
        self._current_queue_size += 1
        self._arrive_times.append(time)
        self._requests.append(n)

    def remove(self, time):
        self._current_queue_size -= 1
        arr_time = self._arrive_times.pop(0)
        wait_time = time - arr_time
        old_cnt = self._avg_waiting_time * self._time_recalcs
        self._time_recalcs += 1
        old_cnt += wait_time
        self._avg_waiting_time = old_cnt / self._time_recalcs

        return self._requests.pop()

    def increase_size(self):
        self._max_queue_size += 1

    def recalc_avg_queue_size(self):
        old_cnt = self._avg_queue_size * self._avg_recalcs
        self._avg_recalcs += 1
        old_cnt += self._current_queue_size
        self._avg_queue_size = old_cnt / self._avg_recalcs


class RequestProcessor(RequestGenerator):
    def __init__(self, generator1, generator2, return_probability):
        super().__init__(generator1)
        self._generator1 = generator1
        self._generator2 = generator2
        self._processed_requests = 0
        self._return_probability = return_probability
        self._reentered_requests = 0
        self._queue = Queue()

    @property
    def processed_requests(self):
        return self._processed_requests

    @property
    def reentered_requests(self):
        return self._reentered_requests

    @property
    def queue(self):
        return self._queue

    def process(self, time):
        if self._queue.current_queue_size > 0:
            self._processed_requests += 1
            n = self._queue.remove(time)
            self.emit_request(time)
            if nr.random_sample() < self._return_probability:
                self._reentered_requests += 1
                self.receive_request(time, n)

            return n

    def receive_request(self, time, n):
        self._queue.add(time, n)
        if self._queue.current_queue_size > self._queue.max_queue_size:
            self._queue.increase_size()

    def next_time_period(self, n):
        if n == 1:
            return self._generator1.generate()
        elif n == 2:
            return self._generator2.generate()


class Model:
    def __init__(self, m1, s1, m2, s2, n1, n2, ret_prob):
        self._generators = [RequestGenerator(GaussDistribution(m1[i], s1[i]), i + 1) for i in range(n1)]
        self._processors = [RequestProcessor(GaussDistribution(m2[0], s2[0]),
                                             GaussDistribution(m2[1], s2[1]), ret_prob) for i in range(n2)]

        for p in self._processors:
            for g in self._generators:
                g.add_receiver(p)

    def time_based_modelling(self, modelling_time, dt):
        generator = self._generators[0]
        processor = self._processors[0]

        gen_period = generator.next_time()
        proc_period = gen_period + processor.next_time()
        current_time = 0
        while current_time < modelling_time:
            #         while current_time < request_count:
            if current_time >= proc_period:
                # print('proc')
                #                 cur_queue = processor.queue.current_queue_size
                # print(processor.processed_requests)
                processor.process(current_time)
                # print(processor.processed_requests)
                #                 if processor.queue.current_queue_size == cur_queue:
                #                     request_count += 1
                if processor.queue.current_queue_size > 0:
                    proc_period += processor.next_time()
                else:
                    proc_period = gen_period + processor.next_time()
            if gen_period <= current_time:
                # print('gen')
                generator.emit_request(current_time)
                gen_period += generator.next_time()
            # print(processor.queue.max_queue_size, processor.queue.current_queue_size)
            current_time += dt
            processor.queue.recalc_avg_queue_size()

        return processor.queue.avg_queue_size, processor.queue.avg_waiting_time

    def time_based_modellingg(self, modelling_time, dt):
        generators = self._generators
        processors = self._processors

        gen_pers = [generators[i].next_time() for i in range(len(generators))]
        proc_pers = [-1 for i in range(len(processors))]

        proc_pers[0] = min(gen_pers) + processors[0].next_time()
        current_time = 0

        while current_time < modelling_time:
            #         while current_time < request_count:
            for i in range(len(processors)):
                if current_time >= proc_pers[i] >= 0:
                    n = processors[i].process(current_time)

                    if processors[i].queue.current_queue_size > 0:
                        proc_pers[i] += processors[i].next_time_period(n)
                    else:
                        proc_pers[i] = -1

            for i in range(len(generators)):
                if gen_pers[i] <= current_time:
                    proc = generators[i].emit_request(current_time)

                    proc_i = processors.index(proc)

                    if proc_pers[proc_i] == -1:
                        proc_pers[proc_i] = gen_pers[i] + processors[proc_i].next_time()

                    gen_pers[i] += generators[i].next_time()

            current_time += dt

            for i in range(len(processors)):
                processors[i].queue.recalc_avg_queue_size()

        return processors[0].queue.avg_queue_size, \
               processors[0].queue.avg_waiting_time, \
               processors[0].processed_requests
