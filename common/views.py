# -*- coding: utf-8 -*-

from common.utils import direct_response
from common.forms import TrainingForm, SimulationForm, TopologyForm
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from pylab import plot, subplot, show, savefig
from pylab import xlabel, ylabel, close, title, grid
from sample_manager.sampleset import main
from sample_manager.sampleset import SampleSet
from neurgen.neuralnet import NeuralNet
from pywt import Wavelet, wavedec
from pyevolve import G1DList
from pyevolve import GSimpleGA
from pyevolve import Selectors
from pyevolve import Statistics
import pyevolve
import os


def start(request):
    """
    Options
    """
    
    return direct_response(request, "common/start.html")


@csrf_exempt
def training(request):
    """
    View to load and preview data to the hybrid system
    """
    
    if request.method == 'POST':
        form = TrainingForm(request.POST, request.FILES)
        # TODO: dos capas ocultas de 10 y 2 nodos con 25 iteraciones, óptimo
        if form.is_valid():
            # Obtener el conjunto de entrenamiento, de pruebas y los targets
            sampleset = SampleSet(form.cleaned_data['wavelet'])
            inputs = sampleset.get_learn_set(request.FILES['learn_file'])
            targets = sampleset.get_targets("%s/sets/target_set.txt" % settings.MEDIA_ROOT)
            
            # Crear la red neuronal con la topología elegida
            hidden1 = form.cleaned_data['hidden1_nn']
            hidden2 = form.cleaned_data['hidden2_nn']
            net = NeuralNet()
            net.init_layers(sampleset.get_input_len(), [hidden1, hidden2], 1)
            net.randomize_network()
            net.learnrate = form.cleaned_data['learnrate']
            net.randomize_network()
            net.set_all_inputs(inputs)
            net.set_all_targets(targets)
            net.set_learn_range(0, 79)
            net.set_test_range(79, 89)
            net.layers[1].set_activation_type(form.cleaned_data['activation'])
            output = net.learn(epochs=form.cleaned_data['epochs'],
                               show_epoch_results=True, 
                               random_testing=True)
            # Guardar la red neuronal
            net.save("%s/nn/neuralnet.txt" % settings.MEDIA_ROOT)
            
            # Plot and save the mse
            plot(range(1, len(net.accum_mse) + 1, 1), net.accum_mse)
            xlabel("Iteraciones")
            ylabel('Error')
            grid(True)
            title(u"Mean Squared Error por iteración")
            savefig("%s/img/mse.png" % settings.MEDIA_ROOT)
            close('all')
            
            return direct_response(request, 'common/training.html',
                                   {'form': form,
                                    'output': output})
    else:
        form = TrainingForm()
        try:
            os.remove("%s/img/mse.png" % settings.MEDIA_ROOT)
        except OSError:
            pass

    return direct_response(request, 'common/training.html',
                           {'form': form,
                            'output': False})


@csrf_exempt
def topology(request):
    """
    Get a topology for neural network from genetic algorithms
    """
    
    if request.method == "POST":
        form = TopologyForm(request.POST)
        if form.is_valid():
            genome = G1DList.G1DList(4)
            genome.setParams(rangemin=1, rangemax=10)
            genome.evaluator.set(eval_func)
            ga = GSimpleGA.GSimpleGA(genome)
            ga.selector.set(Selectors.GRouletteWheel)
            ga.setGenerations(form.cleaned_data['generations'])
            ga.terminationCriteria.set(GSimpleGA.ConvergenceCriteria)
            ga.evolve(freq_stats=1)
            best_topology = ga.bestIndividual()
            
            return direct_response(request, "common/topology.html",
                                   {"form": form,
                                    "output": False})

    else:
        form = TopologyForm()
        
    return direct_response(request, "common/topology.html",
                           {"form": form,
                            "output": False})


@csrf_exempt
def simulation(request):
    """
    Simulation over an neural network
    """
    
    if request.method == 'POST':
        form = SimulationForm(request.POST, request.FILES)
        if form.is_valid():
            # Read data
            data = request.FILES['sim_file'].read().split()
            data = [int(point) for point in data]
            
            # Compress data
            wavelet = Wavelet('db4')
            compress_data = wavedec(data, wavelet, level=5)
            compress_data = [i/10 for i in list(compress_data[0])]
            
            # Plot data and compress data
            subplot(121)
            plot(data)
            title("Espectro Raman original")
            grid(True)
            xlabel("Wavenumber")
            ylabel("Intensidad")   
            #margins(300, 600)
            subplot(122)
            plot(compress_data)
            title("Espectro comprimido")
            grid(True)
            xlabel("Wavenumber")
            ylabel("Intensidad")
            #margins(80, 25)
            savefig("%s/img/datavscompress.png" % settings.MEDIA_ROOT)
            close('all')
            
            # Load neural network and simulate with an input
            net = NeuralNet()
            net.load("%s/nn/neuralnetop.txt" % settings.MEDIA_ROOT)
            output = net.simulation(compress_data)
            
            form = SimulationForm()
            
            return direct_response(request, 'common/simulation.html',
                           {'form': form,
                            'result': output[0],})
    else:
        form = SimulationForm()
        try:
            os.remove("%s/img/datavscompress.png" % settings.MEDIA_ROOT)
        except OSError:
            pass

    return direct_response(request, 'common/simulation.html',
                           {'form': form,
                            'result': False})


def samples(request):
    """
    Write the sample set in a file directory and show it
    """
    
    sampleset = SampleSet()
    for i in range(1, 6):
        sampleset.read("%sdata/para%s.txt" % (settings.MEDIA_ROOT, i))
        for j in range(1, 11):
            sampleset.generate_sample(1)
    for i in range(1, 5):
        sampleset.read("%sdata/orina%s.txt" % (settings.MEDIA_ROOT, i))
        for j in range(1, 13):
            sampleset.generate_sample(0)
    sampleset.save_samples()
    
    return direct_response(request, "common/sample_files.html")
    
    
def eval_func(cromo):
    """
    This function evaluates a chromosome in a neural network,
    the number of neurons on hidden layers, learnrate and activation function
    are evaluate.
    """

    net = NeuralNet()
    net.init_layers(22, [cromo[0] % 15, cromo[1] % 10], 1)
    net.randomize_network()
    net.learnrate = float(cromo[2])/10
    net.randomize_network()

    sampleset = SampleSet()
    with open('%s/sets/learn_test_set.txt' % settings.MEDIA_ROOT, "r") as input_file:
        inputs = sampleset.get_learn_set(input_file)
    targets = sampleset.get_targets("%s/sets/target_set.txt" % settings.MEDIA_ROOT)

    net.set_all_inputs(inputs)
    net.set_all_targets(targets)

    net.set_learn_range(0, 79)
    net.set_test_range(79, 89)
    
    act = ['linear', 'sigmoid', 'tanh']

    net.layers[1].set_activation_type(act[(cromo[3] % 4) - 1])
    net.learn(epochs=45, show_epoch_results=False, random_testing=False)
    print u"Fitness: %s\nNeuronas de la capa oculta 1: %s \
            \nNeuronas de la capa oculta 2: %s\nTaza de aprendizaje: %s \
            \nFunción de activación: %s\n -------------------------------------------" % \
            (1 - net.test(), 
             cromo[0] % 15, 
             cromo[1] % 10, 
             float(cromo[2])/10, 
             act[(cromo[3] % 4) - 1])
    
    return 1 - net.test()
    
