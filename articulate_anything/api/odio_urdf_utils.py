import six
import xml.etree.ElementTree as ET
import copy
import inspect
import sys
import os


def eval_macros(string, env):
    to_eval_start = string.find('${')
    if to_eval_start == -1:
        return string
    to_eval_end = string.find('}', to_eval_start)
    if (to_eval_start != to_eval_end):
        res = eval(string[to_eval_start+2:to_eval_end], env)
        string = string[:to_eval_start]+str(res)+string[to_eval_end+1:]
    return eval_macros(string, env)


class ElementMeta(type):
    """ Metaclass for URDF element subclasses """
    _defaults_ = dict(
        required_elements=[],
        allowed_elements=[],
        required_attributes=[],
        allowed_attributes=[],
    )

    def __new__(cls, name, bases, clsdict):
        """
            Populate class attributes from _default_ if they are not explicitly defined
        """
        for k in cls._defaults_:
            if not k in clsdict:
                clsdict[k] = cls._defaults_[k]

        return super(ElementMeta, cls).__new__(cls, name, bases, clsdict)


class NamedElementMeta(ElementMeta):
    """ Many elements have 'name' as a required attribute """
    _defaults_ = dict(ElementMeta._defaults_, required_attributes=["name"])


def instantiate_if_class(subject):
    """ If subject is a type instead of an instance, instantiate it"""
    if type(subject) in [type, ElementMeta, NamedElementMeta]:
        ret = globals()[subject.__name__]()
    else:
        ret = subject
    return ret


def classname(obj):
    """ Return class name for instance or class object """
    obj_type = type(obj)
    if obj_type in [type, ElementMeta, NamedElementMeta]:
        return obj.__name__
    else:
        return obj_type.__name__


def literal_as_str(literal):
    """ Returns value literals as a str and tuple/lists as a space separated str """
    if isinstance(literal, int) or isinstance(literal, float) or isinstance(literal, str):
        return str(literal)
    elif isinstance(literal, tuple) or isinstance(literal, list):
        return " ".join([str(x) for x in literal])


def urdf_to_odio(urdf_string):
    """
        Dump a URDF string to odio DSL representation
    """
    root = ET.fromstring(urdf_string)

    s = xml_to_odio(root)
    return s


def xml_to_odio(root, depth=0):
    """
        Dump an xml ElementTree to the odio DSL representation
    """
    special_names = {}
    s = ""

    name = root.tag
    if (root.tag[0] == '{'):
        name = 'xacro'+root.tag[root.tag.find('}')+1:]

    if name in special_names:
        name = special_names[name]

    s += "\n" + ' '*depth + name.capitalize() + '('

    for tag in root:
        s += xml_to_odio(tag, depth+1) + ','

    if len(root.attrib.items()) < 3:
        space = ""
        if (len(root) > 0):
            s += '\n'+' '*(depth+1)
    else:
        space = '\n'+' '*(depth+1)

    for key, value in root.attrib.items():
        s += space + key + '= "'+value+'",'

    if root.text and root.text.strip() != "":
        s += space + 'xmltext = "'+root.text+'",'

    if s[-1] == ',':
        s = s[:-1]

    if len(root) < 0:
        s += ')'
    else:
        s += space + ')'
        # s+= '\n'+' '*depth +')'
    return s


