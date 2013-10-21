# -*- coding: utf-8 -*-

import codecs
from django.conf import settings
from django.core.mail import EmailMessage, BadHeaderError
from django.http import HttpResponse
from django.shortcuts import render_to_response
from django.template import RequestContext
from django.utils import simplejson
from django.core.paginator import Paginator, InvalidPage, EmptyPage
from random import Random
import string


def direct_response(request, *args, **kwargs):
    """
    Forma resumida de render_to_response, enviando context_instance al template
    """
    kwargs['context_instance'] = RequestContext(request)
    return render_to_response(*args, **kwargs)


def json_response(data):
    """
    Devuelve una respuesta json con la información de data
    """
    return HttpResponse(simplejson.dumps(data), mimetype = 'application/json')


def send_html_mail(subject, html_file, data, from_email, to_emails, files=None):
    """
    Envía un e-mail con contenido html el cual es extraído de un archivo de 
    codificación utf-8 ubicado en MEDIA_URL/html colocando la data correcta, 
    la cúal debe ser una lista, como parámetro opcional se pueden poner archivos
    adjuntos en forma de lista
    """
    html = codecs.open('%shtml/%s' % (settings.MEDIA_ROOT, html_file), "r",
                       "utf-8")
    content = html.read() % data
    html.close()

    try:
        msg = EmailMessage(subject, content, from_email, to_emails)
        msg.content_subtype = "html"
        if files == None:
            pass
        else:    
            for afile in files:
                msg.attach_file(afile)
        msg.send()

    except BadHeaderError:
        return HttpResponse('Se encontró una cabecera de e-mail inválida')
        

def get_paginated(request, object_list, num_items):
    """
    Devuelve una lista paginada de una lista de objetos y el
    número de objetos por página
    """
    paginator = Paginator(object_list, num_items)   
         
    try:
        page = int(request.GET.get('page', '1'))
    except ValueError:
        page = 1        
        
    try:
        lista_paginada = paginator.page(page)
    except (EmptyPage, InvalidPage):
        lista_paginada = paginator.page(paginator.num_pages)
        
    return lista_paginada
    
    
def make_password(length=8):
    """
    Devuelve una cadena aleatoria de tamaño length
    """
    return ''.join(Random().sample(string.letters+string.digits, length))

