#!/usr/bin/env python
###############################################################################
# Top contributors (to current version):
#   Yoni Zohar, Abdalrhman Mohamed, Alex Ozdemir
#
# This file is part of the cvc5 project.
#
# Copyright (c) 2009-2022 by the authors listed in the file AUTHORS
# in the top-level source directory and their institutional affiliations.
# All rights reserved.  See the file COPYING in the top-level source
# directory for licensing information.
# #############################################################################
#
# Utility Methods, translated from examples/api/utils.h
##

import cvc5
from cvc5 import Kind
from imgman_dsl import *

# Get the string version of define-fun command.
# @param f the function to print
# @param params the function parameters
# @param body the function body
# @return a string version of define-fun


def define_fun_to_string(f, params, body):
    sort = f.getSort()
    if sort.isFunction():
        sort = f.getSort().getFunctionCodomainSort()
    result = "(define-fun " + str(f) + " ("
    for i in range(0, len(params)):
        if i > 0:
            result += " "
        result += "(" + str(params[i]) + " " + str(params[i].getSort()) + ")"
    result += ") " + str(sort) + " " + str(body) + ")"
    return result

constants = {"MouthOpen": MouthOpen(), 
            "EyesOpen": EyesOpen(), 
            "BelowAge18": BelowAge(18), 
            "Smile": IsSmiling(), 
            "IsPrice": IsPrice(),
            "IsPhoneNumber": IsPhoneNumber(), 
            "TypeFace": IsFace(), 
            "TypeText": IsText(),
            "GetLeft": GetLeft(), 
            "GetRight": GetRight(), 
            "GetAbove": GetAbove(), 
            "GetBelow": GetBelow(), 
            "GetNext": GetNext(), 
            "GetPrev": GetPrev(), 
            "GetChildren": GetContains(), 
            "GetParents": GetIsContained()}

def constant_to_prog(s, id_to_text):
    if s in constants:
        return constants[s]
    if s.startswith("Name"):
        return IsObject(s[4:].replace('1', ' '))
    if s.startswith("Text"):
        return MatchesWord(id_to_text[int(s[4:])])
    if s.startswith("Index"):
        return GetFace(int(s[5:]))

def expression_to_imgman(body, id_to_pred, id_to_text):
    if str(body[0]).startswith('match'):
        return constant_to_prog(id_to_pred[int(str(body[0])[5:])], id_to_text)
    elif str(body.getKind()) == "Kind.SET_UNION":
        return Union([expression_to_imgman(body[0], id_to_pred, id_to_text), expression_to_imgman(body[1], id_to_pred, id_to_text)])
    elif str(body.getKind()) == "Kind.SET_INTER":
        return Intersection([expression_to_imgman(body[0], id_to_pred, id_to_text), expression_to_imgman(body[1], id_to_pred, id_to_text)])
    elif str(body.getKind()) == "Kind.SET_MINUS":
        return Complement(expression_to_imgman(body[1], id_to_pred, id_to_text))
    else:
        return None


# Print solutions for synthesis conjecture to the standard output stream.
# @param terms the terms for which the synthesis solutions were retrieved
# @param sols the synthesis solutions of the given terms


def print_synth_solutions(terms, sols):
    result = "(\n"
    for i in range(0, len(terms)):
        params = []
        body = sols[i]
        if sols[i].getKind() == Kind.LAMBDA:
            params += sols[i][0]
            body = sols[i][1]
        result += "  " + define_fun_to_string(terms[i], params, body) + "\n"
    result += ")"
    print(result)
