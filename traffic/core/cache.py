import json
from collections import UserDict
from pathlib import Path


class property_cache(object):
    """Computes attribute value and caches it in the instance.
    Python Cookbook (Denis Otkidach)
    https://stackoverflow.com/users/168352/denis-otkidach

    This decorator allows you to create a property which can be computed once
    and accessed many times. Sort of like memoization, but by instance.

    """

    def __init__(self, method):
        # record the unbound-method and the name
        self.method = method
        self.name = method.__name__
        self.__doc__ = method.__doc__
        self.__annotations__ = method.__annotations__

    def __get__(self, instance, cls):
        # self: <__main__.cache object at 0xb781340c>
        # inst: <__main__.A object at 0xb781348c>
        # cls: <class '__main__.A'>
        if instance is None:
            # instance attribute accessed on class, return self
            # You get here if you write `A.bar`
            return self
        # compute, cache and return the instance's attribute value
        result = self.method(instance)
        # setattr redefines the instance's attribute
        setattr(instance, self.name, result)
        return result


class Cache(UserDict):
    def __init__(self, cachedir: Path) -> None:
        self.cachedir = cachedir
        if not self.cachedir.exists():
            self.cachedir.mkdir(parents=True)
        super().__init__()

    def __missing__(self, hashcode: str):
        filename = self.cachedir / f"{hashcode}.json"
        if filename.exists():
            response = json.loads(filename.read_text())
            # Do not overwrite the file!
            super().__setitem__(hashcode, response)
            return response

    def __setitem__(self, hashcode: str, data):
        super().__setitem__(hashcode, data)
        filename = self.cachedir / f"{hashcode}.json"
        with filename.open("w") as fh:
            fh.write(json.dumps(data))
