from django.conf.urls.defaults import *

from os.path import dirname
basedir = dirname(__file__)

media = '%s/media/' % basedir

urlpatterns = patterns('common.views',
    url(r'^$', 'start', name='start'),
    url(r'^training$', 'training', name='training'),
    url(r'^topology$', 'topology', name='topology'),
    url(r'^simulation$', 'simulation', name='simulation'),
    url(r'^samples$', 'samples', name='samples'),
)

urlpatterns += patterns('',
    # media url
    (r'^media/(?P<path>.*)$', 'django.views.static.serve',
        {'document_root': media, 'show_indexes': True}),
)
