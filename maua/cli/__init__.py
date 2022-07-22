import importlib


def main_function(name):
    def delayed_execution(args):
        importlib.import_module(name).main(args)

    return delayed_execution
