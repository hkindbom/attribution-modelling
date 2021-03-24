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
    def __init__(self, id, init_click_prob=0.02, init_conv_prob=0.02, max_prob=0.85, moment_click_fact=1, moment_conv_fact=10):
        self.id = id
        self.click_prob = init_click_prob # probability of clicking when exposed
        self.conv_prob = init_conv_prob # probability of converting when clicked
        self.max_prob = max_prob
        self.moment_click_fact = moment_click_fact
        self.moment_conv_fact = moment_conv_fact
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

        self.click_prob = min(self.max_prob, self.click_prob)
        self.conv_prob = min(self.max_prob, self.conv_prob)

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

    def get_ratio_adjusted(self, ratio_maj_min_class, persons_dict):
        nr_per_class = np.zeros(2, dtype=int)
        for person_id in persons_dict:
            nr_per_class[persons_dict[person_id]['label']] += 1
        min_class = np.argmin(nr_per_class)
        nr_min = nr_per_class.min()
        nr_max = int(nr_min * ratio_maj_min_class)
        nr_pos_max = nr_min if min_class == 1 else nr_max
        nr_neg_max = nr_min if min_class == 0 else nr_max
        return nr_pos_max, nr_neg_max

    def get_adjusted_nr_samples(self, persons_dict, ratio_maj_min_class, nr_pos_max, nr_neg_max):
        if nr_pos_max is None or nr_neg_max is None:
            nr_pos_max, nr_neg_max = self.get_ratio_adjusted(ratio_maj_min_class, persons_dict)
        count_pos = 0
        count_neg = 0
        filtered_persons_dict = {}
        for person_id in persons_dict:
            if persons_dict[person_id]['label'] and count_pos < nr_pos_max:
                count_pos += 1
                filtered_persons_dict[person_id] = persons_dict[person_id]
            if not persons_dict[person_id]['label'] and count_neg < nr_neg_max:
                count_neg += 1
                filtered_persons_dict[person_id] = persons_dict[person_id]

        print('Nr converted sim: ', count_pos, 'Nr non-converted sim: ', count_neg)
        if nr_pos_max + nr_neg_max != len(filtered_persons_dict):
            print('WARNING! Nr simulated samples don\'t match real data')
        return filtered_persons_dict

    def get_data_dict_format(self, ratio_maj_min_class, nr_pos, nr_neg):
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
        return self.get_adjusted_nr_samples(persons_dict, ratio_maj_min_class, nr_pos, nr_neg)

    def show_results(self):
        for person in self.persons:
            print('person', person.id, ' signed ? ', person.converted, ' costs: ', person.costs)
        print('nr conversions: ', self.nr_conversions)
        print('tot spend: ', self.tot_spend)

    def handle_event(self, channel, person_id):
        person = self.persons[person_id]
        if person.converted:
            return

        click_prob = min(person.max_prob, person.click_prob + person.moment_click_fact * channel.click_prob_inc)
        clicked_ch = np.random.choice([True, False], p=[click_prob, 1 - click_prob])
        if clicked_ch:
            conv_prob = min(person.max_prob, person.conv_prob + person.moment_conv_fact * channel.conv_prob_inc)
            converted = np.random.choice([True, False], p=[conv_prob, 1 - conv_prob])
            person.register_click(channel, self.current_time, self.ch_interact)
            self.tot_spend += channel.cpc
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
        int_factor = 0.5/3
        self.channels.append(Channel(0, '(direct) / (none)', 0, 0.055, 0.055, 0.15*int_factor))
        self.channels.append(Channel(1, 'adtraction / affiliate', 1, 0.065, 0.065, 0.1*int_factor))
        self.channels.append(Channel(2, 'facebook / ad', 2, 0.005, 0.005, 0.15*int_factor))
        self.channels.append(Channel(3, 'google / cpc', 3, 0.055, 0.055, 0.1*int_factor))
        self.channels.append(Channel(4, 'google / organic', 4, 0.055, 0.055, 0.1*int_factor))
        self.channels.append(Channel(5, 'mecenat / partnership', 5, 0.07, 0.07, 0.15*int_factor))
        self.channels.append(Channel(6, 'newsletter / email', 6, 0.075, 0.075, 0.1*int_factor))
        self.channels.append(Channel(7, 'snapchat / ad', 7, 0.0, 0.0, 0.3*int_factor))
        self.channels.append(Channel(8, 'studentkortet / partnership', 8, 0.065, 0.065, 0.1*int_factor))
        self.channels.append(Channel(9, 'tiktok / ad', 9, 0.0, 0.0, 0.2*int_factor))
        self.ch_interact = np.ones((10, 10)).tolist()

if __name__ == '__main__':
    cohort_size = 20
    sim_time = 30

    sim = Simulator(cohort_size, sim_time)
    sim.run_simulation()
    sim.show_results()
    data_dict = sim.get_data_dict_format(2, 2)

