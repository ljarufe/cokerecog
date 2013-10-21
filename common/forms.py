# -*- coding: utf-8 -*-

from django import forms


class TrainingForm(forms.Form):
    """
    Form to load training and testing data to the neural network
    """
    learn_file = forms.FileField(label=u"Conjunto de entrenamiento")
    hidden1_nn = forms.IntegerField(label=u"Neuronas de la primera capa oculta",
                                    initial=10)
    hidden2_nn = forms.IntegerField(label=u"Neuronas de la segunda capa oculta",
                                    initial=2)
    WAVELET_CHOICES = (
        (u'db4', u'Daubechies 4'),
        (u'db3', u'Daubechies 3'),
        (u'db2', u'Daubechies 2'),
        (u'db1', u'Daubechies 1'),
        (u'haar', u'Haar'),
    )
    wavelet = forms.CharField(widget=forms.Select(choices=WAVELET_CHOICES),
                              label=u"Transformada wavelet")
    ACTIVATION_CHOICES = (
        (u'tanh', u'Tangente hiperbólica'),
        (u'sigmoid', u'Sigmoide'),
        (u'linear', u'Lineal'),
    )
    activation = forms.CharField(widget=forms.Select(choices=ACTIVATION_CHOICES),
                                 label=u"Función de activación")
    epochs = forms.IntegerField(label=u"Iteraciones", initial=50)
    learnrate = forms.FloatField(label=u"Taza de aprendizaje", initial=0.2)
    

class TopologyForm(forms.Form):
    """
    Form to set the genetic algorithm to find a neural network topology
    """
    generations = forms.IntegerField(label="Generaciones", initial=50)


class SimulationForm(forms.Form):
    """
    Form to load simulation data
    """
    sim_file = forms.FileField(label="Muestra para la simulación")

