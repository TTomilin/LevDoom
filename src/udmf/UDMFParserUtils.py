import os
import sys
from typing import List

import omg

PROGRAM_NAME = os.path.basename(sys.argv[0]).rsplit(".", 1)[0]


class NoMapInWADException(Exception):
	"""No UDMF maps found in WAD"""
	pass


class NoMapHeaderForTextmapException(Exception):
	"""TEXTMAP has no corresponding map header entry"""
	pass


class MapEntry:
	"""
	A class that encapsulates a WadIO reference, a corresponding WadIO Entry (TEXTMAP entry), and a name of this map.
	"""

	def __init__(self, wadio: omg.WadIO, entry, name: str):
		self.wadio: omg.WadIO = wadio
		self.entry = entry
		self.name: str = name
		pass

	@property
	def textmap(self) -> str:
		return self.get_textmap()

	def get_textmap(self) -> str:
		lastoffs = self.wadio.basefile.tell()

		self.wadio.basefile.seek(self.entry.ptr)
		textmap = self.wadio.basefile.read(self.entry.size).decode("utf-8")
		self.wadio.basefile.seek(lastoffs)

		return textmap


def next_map_entry_from_wad(wadio: omg.WadIO):
	textmap_entry_nums: List[int] = wadio.multifind("TEXTMAP")

	if len(textmap_entry_nums) == 0:
		raise NoMapInWADException(
			"\"" + os.path.basename(wadio.basefile.name) + "\"" + " does not contain any UDMF maps")

	for entrynum in textmap_entry_nums:
		if entrynum - 1 < 0 and wadio.entries[entrynum - 1].size != 0:
			raise NoMapHeaderForTextmapException(
				"\"" + wadio.basefile.name + "\""
				+ " contains a TEXTMAP that does not have a corresponding map header")

		map_entry_name: str = os.path.basename(wadio.basefile.name) + "/" + wadio.entries[entrynum - 1].name

		return MapEntry(wadio, wadio.entries[entrynum], map_entry_name)
