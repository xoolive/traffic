import logging


class PluginProvider:
    """
    Mount point for plugins which refer to actions that can be performed.
    Plugins implementing this reference should provide load_plugin method.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if not hasattr(cls, "plugins"):
            # This branch only executes when processing the mount point itself.
            # So, since this is a new plugin type, not an implementation, this
            # class shouldn't be registered as a plugin. Instead, it sets up a
            # list where plugins can be registered later.
            cls.plugins = []
        # This must be a plugin implementation, which should be registered.
        # Simply appending it to the list is all that's needed to keep
        # track of it later.
        cls.plugins.append(cls)

    def __init__(self) -> None:
        self.title = self.__class__.__name__
        logging.debug(f"Initialize plugin {self.title}")

    def load_plugin(self) -> None:
        pass
