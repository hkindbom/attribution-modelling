import heapq
import numpy as np

class Channel:
    def __init__(self, index, name, cpc, click_prob_inc, conv_prob_inc, exposure_intensity):
        self.index = index
        self.name = name
        self.cpc = cpc
        self.click_prob_inc = click_prob_inc
        self.conv_prob_inc = conv_prob_inc
        self.exposure_intensity = exposure_intensity

class Person:
    def __init__(self, id, init_click_prob=0.01, init_conv_prob=0.02):
        self.id = id
        self.click_prob = init_click_prob # probability of clicking when exposed
        self.conv_prob = init_conv_prob # probability of converting when clicked
        self.converted = False
        self.clicked_channels = []
        self.click_times = []
        self.costs = []

    def register_click(self, channel, time, ch_interact):
        self.clicked_channels.append(channel)
        self.click_times.append(time)
        self.costs.append(channel.cpc)
        self.upd_probas(ch_interact)

    def upd_probas(self, ch_interact):
        channel_last = self.clicked_channels[-1]
        if len(self.clicked_channels) == 1:
            self.click_prob += channel_last.click_prob_inc
            self.conv_prob += channel_last.conv_prob_inc
        else:
            channel_pre = self.clicked_channels[-2]
            self.click_prob += ch_interact[channel_pre.index][channel_last.index] * channel_last.click_prob_inc
            self.conv_prob += ch_interact[channel_pre.index][channel_last.index] * channel_last.conv_prob_inc

        self.click_prob = min(1, self.click_prob)
        self.conv_prob = min(1, self.conv_prob)

class Simulator:
    def __init__(self, cohort_size, sim_time):
        self.events = []
        self.persons = []
        self.channels = []
        self.ch_interact = []
        self.cohort_size = cohort_size
        self.tot_spend = 0
        self.nr_conversions = 0
        self.current_time = 0
        self.sim_time = sim_time
        self.init_simulator()

    def run_simulation(self):
        while self.current_time < self.sim_time and self.nr_conversions < self.cohort_size:
            next_event = heapq.heappop(self.events)
            event_time = next_event[0]
            event_ch = next_event[1]
            event_person_id = next_event[2]
            self.current_time = event_time
            self.handle_event(event_ch, event_person_id)
            self.create_new_event(event_ch, event_person_id)
        print('Simulation done')


    def get_data_dict_format(self):
        persons_dict = {}
        for person in self.persons:
            if len(person.click_times) > 0:
                persons_dict[person.id] = {}
                persons_dict[person.id]['label'] = int(person.converted)
                persons_dict[person.id]['cost'] = person.costs
                persons_dict[person.id]['session_times'] = person.click_times
                persons_dict[person.id]['session_channels'] = []
                for channel in person.clicked_channels:
                    persons_dict[person.id]['session_channels'].append(channel.index)
        return persons_dict

    def show_results(self):
        for person in self.persons:
            print('person', person.id, ' signed ? ', person.converted, ' costs: ', person.costs)
        print('nr conversions: ', self.nr_conversions)
        print('tot spend: ', self.tot_spend)

    def handle_event(self, channel, person_id):
        person = self.persons[person_id]
        if person.converted:
            return

        clicked_ch = np.random.choice([True, False], p=[person.click_prob, 1 - person.click_prob])
        if clicked_ch:
            person.register_click(channel, self.current_time, self.ch_interact)
            self.tot_spend += channel.cpc

            converted = np.random.choice([True, False], p=[person.conv_prob, 1 - person.conv_prob])
            if converted:
                person.converted = True
                self.nr_conversions += 1

        self.persons[person_id] = person

    def create_new_event(self, channel, person_id):
        if not self.persons[person_id].converted:
            self.add_channel_event(person_id, channel)

    def init_simulator(self):
        self.create_channels()
        for person_id in range(self.cohort_size):
            person = Person(person_id)
            self.persons.append(person)
            self.add_channel_events(person.id)

    def add_channel_event(self, person_id, channel):
        exposure_time = self.get_exposure_time(channel.exposure_intensity)
        heapq.heappush(self.events, (exposure_time, channel, person_id))

    def add_channel_events(self, person_id):
        for channel in self.channels:
            self.add_channel_event(person_id, channel)

    def get_exposure_time(self, intensity):
        return self.current_time + np.random.exponential(1/intensity)

    def get_ch_idx_maps(self):
        ch_to_idx = {}
        idx_to_ch = {}
        for channel in self.channels:
            ch_to_idx[channel.name] = channel.index
            idx_to_ch[channel.index] = channel.name
        return ch_to_idx, idx_to_ch

    def create_channels(self):
        self.channels.append(Channel(0, 'fb', 10, 0.2, 0.4, 0.2))
        self.channels.append(Channel(1, 'google', 15, 0.0, 0.0, 0.3))
        self.ch_interact = [[1, 1.1],
                            [1.05, 1]]

if __name__ == '__main__':
    cohort_size = 10
    sim_time = 100

    sim = Simulator(cohort_size, sim_time)
    sim.run_simulation()
    sim.show_results()
    data_dict = sim.get_data_dict_format()

