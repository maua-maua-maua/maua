import importlib


def main_function(name):
    def delayed_execution():
        importlib.import_module(name).main()

    return delayed_execution
