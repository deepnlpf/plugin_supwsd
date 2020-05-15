#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re

from os import path

from deepnlpf.core.iplugin import IPlugin

class Plugin(IPlugin):

    def __init__(self, id_pool, lang, document, pipeline):
        self._id_pool = id_pool
        self._lang = lang
        self._document = document
        self._pipeline = pipeline

        self.HERE = path.abspath(path.dirname(__file__))
        
    def run(self):
        from deepnlpf.core.boost import Boost
        annotation = Boost().multithreading(self.wrapper, self._document['sentences'])
        return annotation

    def wrapper(self, sentence):
        from deepnlpf.core.execute import Execute

        jar_file = self.HERE+'/supwsd-pocket.jar'
        language = 'en'
        model = 'semcor_omsti'
        workspace = self.HERE

        args = [sentence, language, model, workspace]

        doc = Execute().run_java(jar_file, *args)

        doc = doc.decode("utf-8")
        doc = doc.split("\n") # generate list.
        doc.pop(0) # remove sentence inicial
        doc = "".join(doc)

        pattern = re.compile(r'\s+')
        doc = re.sub(pattern, '', doc)
        
        print(doc)
        print(type(doc))
        
        return {doc}

    def out_format(self, doc):
        pass
