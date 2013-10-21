"""
This package fills a poblation of samples to learn, test and simulation
"""

import random
from django.conf import settings
from pywt import Wavelet, wavedec


class SampleSet():
    """    
    A poblation of samples
    """
    
    def __init__(self, transform=None):
        """
        Constructor
        """
        
        self.init_sample = None
        self.learn_set = []
        self.test_set =[]
        self.targets = []
        self.sample_len = 0
        
        if transform == None:
            transform = 'db4'
        self.wavelet = Wavelet(transform)
    
    def read(self, file_name):
        """
        This function read the txt file and put this in a list
        """
        
        sample = self._read_file(file_name)
        self.init_sample = [int(point) for point in sample]
        
    def _read_file(self, file_name):
        """
        This function reads a document and returns a list
        """
        
        with open(file_name, "r") as doc:
            return doc.read().split()
        
    
    def generate_sample(self, target):
        """
        Get samples from init_sample
        """
        
        new_sample = [point+random.randint(-4, 4) for point in self.init_sample]
        if (len(self.learn_set) + len(self.test_set)) % 5 == 0:
            self.test_set.append(new_sample)
            self.targets.append(target)
        else:
            self.learn_set.append(new_sample)
            self.targets.insert(len(self.learn_set), target)
            
        
    def save_samples(self):
        """
        Save all the generated samples in a samples.txt file
        """
        
        doc_count = sim_count = 1
        with open("%ssets/learn_test_set.txt" % settings.MEDIA_ROOT, "w") as lt_file:
            for sample in self.learn_set:
                if doc_count % 9 == 0:
                    with open("%ssets/sim_sample%s.txt" % (settings.MEDIA_ROOT, sim_count), "w") as sim_doc:
                        self._write_sample(sim_doc, sample)
                    sim_count += 1
                    self.targets.pop(doc_count-1)
                else:
                    sample = wavedec(sample, self.wavelet, level=5)
                    sample = [i/10 for i in list(sample[0])]
                    self._write_sample(lt_file, sample)
                    lt_file.write(";")
                doc_count += 1
            for sample in self.test_set:
                sample = wavedec(sample, self.wavelet, level=5)
                sample = [i/10 for i in list(sample[0])]
                self._write_sample(lt_file, sample)
                lt_file.write(";")
        with open("%ssets/target_set.txt" % settings.MEDIA_ROOT, "w") as target_doc:
            self._write_sample(target_doc, self.targets)
                
    def _write_sample(self, outfile, sample):
        """
        Writes a sample in the outfile
        """
        
        for point in sample:
            outfile.write("%s " % point)
        
    def get_learn_set(self, inputfile):
        """
        Get a learn set from a file
        """
        
        raw_data = inputfile.read().split(";")
        samples = []
        for raw_sample in raw_data:
            sample = raw_sample.split()
            samples.append([float(point) for point in sample])
        samples.pop()
        self.sample_len = len(samples[0])

        return samples
        
    def get_targets(self, file_name=None):
        """
        Get a target set for the sample data
        """
        
        if file_name:
            self.targets = self._read_file(file_name)

        return [[float(target)] for target in self.targets]
        
    def get_input_len(self):
        """
        This function returns the len of a compress input
        """
        
        return self.sample_len


def main():
    sampleset = SampleSet()
    for i in range(1, 6):
        sampleset.read("../media/data/para%s.txt" % i)
        for j in range(1, 11):
            sampleset.generate_sample(1.0)
    for i in range(1, 5):
        sampleset.read("../media/data/orina%s.txt" % i)
        for j in range(1, 13):
            sampleset.generate_sample(0.0)
    sampleset.save_samples()


if __name__ == "__main__":
    main()

