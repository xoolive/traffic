from typing import Optional, Type, TypeVar
from xml.dom import minidom
from xml.etree import ElementTree

T = TypeVar("T", bound="B2BReply")


class B2BReply:
    def __init__(self, *args, **kwargs) -> None:
        self.reply: Optional[ElementTree.Element] = None

    @classmethod
    def fromET(cls: Type[T], tree: ElementTree.Element) -> T:
        instance = cls()
        instance.reply = tree
        return instance

    def __str__(self) -> str:
        if self.reply is None:
            return "[empty]"
        s = ElementTree.tostring(self.reply)
        return minidom.parseString(s).toprettyxml(indent="  ")

    def __repr__(self) -> str:
        res = str(self)
        if len(res) > 1000:
            return res[:1000] + "..."
        return res
