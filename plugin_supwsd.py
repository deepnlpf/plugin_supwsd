#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re, json

from os import path

from deepnlpf.core.iplugin import IPlugin
from deepnlpf.core.execute import Execute
from deepnlpf.core.boost import Boost


class Plugin(IPlugin):
    def __init__(self, document, pipeline):
        self._document = document
        self._pipeline = pipeline

        self.HERE = path.abspath(path.dirname(__file__))

    def run(self):
        doc = Boost().multithreading(self.wrapper, self._document)
        return self.out_format(doc)

    def wrapper(self, sentence):
        jar_file = self.HERE + "/supwsd-pocket.jar"
        language = "en"
        model = "semcor_omsti"
        workspace = self.HERE

        args = [sentence, language, model, workspace]

        doc = Execute().run_java(jar_file, *args)

        doc = doc.decode("utf-8")
        doc = doc.split("\n")  # generate list.
        doc.pop(0)  # remove sentence inicial
        doc = "".join(doc)

        pattern = re.compile(r"\s+")
        doc = re.sub(pattern, "", doc)
        return doc

    def out_format(self, annotation):
        doc_formated = list()

        for index, sent in enumerate(annotation):
            text = list()
            token_annotation = list()

            for item in json.loads(sent):
                text.append(item['token']['word'])
            
            data = {}
            data['_id'] = index + 1
            data['text'] = " ".join(text)
            data['annotation'] = json.loads(sent)
            
            doc_formated.append(data)

        return doc_formated
