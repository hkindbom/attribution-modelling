import heapq
import numpy as np
""" To include
- create nr of people
- each channel increases persons conversion intensity, a cpc and click intensity (determined by budget click rate per kr)

fb: ClickI 0.2, ConvI 0.3, day 2
google: ClickI 0.1, ConvI 0.4, day 6

person i
. clickI, convI, channels, times, label, active_status
- clicks fb and then google
- persons click intensity = ch_interaction[fb, google](0.2 + 0.1)
- persons conv intensity =  ch_interaction[fb, google]*(convIfb + convIgoogle)

fb, google, google, snap

channel conv interaction matrix
        fb  google
fb      1   2
google  0.5 1

channel click interaction matrix
        fb  google
fb      1   3
google  0.5 1

"""
class Channel:
    def __init__(self, name, cpc, click_intensity_inc, conv_intensity_inc, exposure_intensity):
        self.name = name
        self.cpc = cpc
        self.click_intensity_inc = click_intensity_inc
        self.conv_intensity_inc = conv_intensity_inc
        self.exposure_intensity = exposure_intensity

class Person:
    def __init__(self, id):
        self.id = id
        self.click_prob = 0.1
        self.conv_prob = 0.2
        self.seen_channels = []
        self.clicked_channels = []
        self.converted = False
        self.conv_value = None

class Simulator:
    def __init__(self, population_size, sim_time):
        self.events = []
        self.persons = []
        self.channels = []
        self.population_size = population_size
        self.tot_spend = 0
        self.current_time = 0
        self.sim_time = sim_time
        self.create_channels()
        self.create_events()

    def run_simulation(self):
        print('Running simulation')
        while self.current_time < self.sim_time:
            next_event = heapq.heappop(self.events)
            event_time = next_event[0]
            event_ch = next_event[1]
            event_person_id = next_event[2]
            self.current_time = event_time
            self.handle_event(event_ch, event_person_id)

            if len(self.events) == 0:
                self.create_events()

    def handle_event(self, channel, person_id):
        pass


    def create_channels(self):
        self.channels.append(Channel('fb', 10, 0.2, 0.1, 0.1))
        self.channels.append(Channel('google', 15, 0.3, 0.2, 0.4))

    def create_events(self):
        for person_idx in range(self.population_size):
            person = Person(person_idx)
            self.persons.append(person)
            for channel in self.channels:
                exposure_time = self.get_exposure_time(channel.exposure_intensity)
                heapq.heappush(self.events, (exposure_time, channel, person.id))

    def get_exposure_time(self, intensity):
        return self.current_time + np.random.exponential(1/intensity)


if __name__ == '__main__':
    population_size = 10
    sim_time = 25
    sim = Simulator(population_size, sim_time)
    sim.run_simulation()

