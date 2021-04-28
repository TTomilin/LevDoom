from enum import IntEnum
from typing import List, Dict, Union, Optional


class LexicographicalException(Exception):
	pass


class UnexpectedCharacterException(LexicographicalException):
	"""Unexpected character"""
	pass


class MalformedNumberException(LexicographicalException):
	"""Malformed number"""
	pass


class UnterminatedStringConstantException(LexicographicalException):
	"""Unterminated string constant"""
	pass


class EToken(IntEnum):
	NONE = 0,
	STATEMENT_END = 1,
	STRING = 2,
	FLOAT = 4,
	INTEGER = 8,
	IDENTIFIER = 16,
	ASSIGNMENT_OP = 32,
	BLOCK_START = 64,
	BLOCK_END = 128,
	KEYWORD_TRUE = 256,
	KEYWORD_FALSE = 512,


EToken.BOOLEAN = EToken.KEYWORD_TRUE | EToken.KEYWORD_FALSE
EToken.NUMBER = EToken.INTEGER | EToken.FLOAT
EToken.VALUE = EToken.BOOLEAN | EToken.STRING | EToken.NUMBER


class Token:
	def __init__(self, toktype: EToken, line: int, text: str):
		self._type: EToken = toktype
		self._line: int = line
		self._text: str = text
		pass

	@property
	def type(self) -> EToken:
		return self._type

	@property
	def line(self) -> int:
		return self._line

	@property
	def text(self) -> str:
		return self._text

	@property
	def value(self):
		return self.get_value()

	def get_value(self):
		if self.type == EToken.INTEGER:
			return int(self.text)
		elif self.type == EToken.FLOAT:
			return float(self.text)
		elif self.type == EToken.STRING:
			return self.text
		elif self.type == EToken.KEYWORD_TRUE:
			return True
		elif self.type == EToken.KEYWORD_FALSE:
			return False


class Lexer:
	def __init__(self, textmap: str):
		self._textmap: str = textmap
		self._src: int = 0
		self._line = 1
		pass

	def __iter__(self):
		return self

	@property
	def line_str(self) -> str:
		return "At line %d" % self._line

	def __next__(self) -> Optional[Token]:
		token_type: EToken
		token_text: str = str()

		self._eat_whitespace()

		if self._current() == "/" and self._peek() == "*":
			while True:
				self._getc()
				if self._current() == "*" and self._peek() == "/":
					self._getc()
					self._getc()
					break
			self._eat_whitespace()

		if self._current() == "/" and self._peek() == "/":
			while self._current() != "\n":
				self._getc()
			self._eat_whitespace()

		if self._current().isalpha() or self._current() == "_":
			token_type = EToken.IDENTIFIER

			while self._current().isalnum() or self._current() == "_":
				token_text += self._getc()

			if token_text.lower() == "true":
				token_type = EToken.KEYWORD_TRUE
			elif token_text.lower() == "false":
				token_type = EToken.KEYWORD_FALSE

		elif self._current() == "=":
			token_type = EToken.ASSIGNMENT_OP
			token_text = self._getc()

		elif self._current() == "\"":
			token_type = EToken.STRING

			self._getc()
			while self._current() != "\"":
				if self._current() == "\n":
					raise UnterminatedStringConstantException("%s: unterminated string constant" % self.line_str)
				token_text += self._getc()
			self._getc()

		elif self._current() == ";":
			token_type = EToken.STATEMENT_END
			token_text = self._getc()

		elif self._current() == "{":
			token_type = EToken.BLOCK_START
			token_text = self._getc()

		elif self._current() == "}":
			token_type = EToken.BLOCK_END
			token_text = self._getc()

		elif self._current().isdecimal() or self._current() == "." or self._current() == "-":
			# while self._current().isdecimal() or self._current() == "." or self._current() == "-":
			while True:
				token_text += self._getc()
				if (not self._current().isdecimal()) and self._current() != ".":
					break
			try:
				int(token_text)
				token_type = EToken.INTEGER
			except ValueError:
				try:
					float(token_text)
					token_type = EToken.FLOAT
				except ValueError:
					raise MalformedNumberException("%s: malformed number \"%s\"" % (self.line_str, token_text))

		else:
			raise UnexpectedCharacterException("%s: unexpected character \"%s\"" % (self.line_str, self._current()))

		return Token(token_type, self._line, token_text)

	def _eat_whitespace(self):
		while self._current().isspace():
			self._getc()
		pass

	def _current(self) -> str:
		if self._src >= len(self._textmap):
			raise StopIteration

		return self._textmap[self._src]

	def _getc(self) -> str:
		if self._src >= len(self._textmap):
			raise StopIteration

		current: str = self._current()

		if current == "\n":
			self._line += 1

		self._src += 1

		return current

	def _peek(self) -> Optional[str]:
		if self._src + 1 >= len(self._textmap):
			return None

		return self._textmap[self._src + 1]


class ParsingException(Exception):
	pass


class UnexpectedTokenException(ParsingException):
	"""Unexpected token"""
	pass


class AssignmentExpr:
	def __init__(self, key: str, value):
		self._key = key
		self._value = value
		pass

	@property
	def key(self) -> str:
		return self._key

	@property
	def value(self):
		return self._value

	def __repr__(self) -> str:
		return "%s: %s" % (self._key, self._value)


class BlockExpr:
	def __init__(self, identifier: str, statements: List[AssignmentExpr] = None):
		self._identifier = identifier
		self._statements = statements
		pass

	@property
	def identifier(self) -> str:
		return self._identifier

	@property
	def statements(self) -> List[AssignmentExpr]:
		return self._statements

	def as_dictionary(self):
		result: dict = {}
		for s in self._statements:
			result[s.key] = s.value
		return result

	def __repr__(self) -> str:
		s: str = ""
		for ass in self._statements:
			s += str(ass) + ", "
		return s


class EParserState(IntEnum):
	NORMAL = 0,
	BLOCK = 1


class Parser:
	def __init__(self, textmap: str):
		self._lexer: Lexer = Lexer(textmap)
		self._curtok: Optional[Token] = next(self._lexer)
		self._state = EParserState.NORMAL
		pass

	def _get_next_token(self) -> Token:
		try:
			self._curtok = next(self._lexer)
		except StopIteration:
			self._curtok = None
		return self._curtok

	def __iter__(self):
		return self

	def __next__(self) -> Optional[Union[AssignmentExpr, BlockExpr]]:
		if self._curtok is None:
			raise StopIteration

		if self._curtok.type == EToken.IDENTIFIER:
			return self._parse_identifier_expr()

		else:
			raise UnexpectedTokenException(
				"%s: Expected identifier, got \"%s\"" % (self._lexer.line_str, self._curtok.text))
		pass

	def _parse_identifier_expr(self) -> Union[AssignmentExpr, BlockExpr]:
		idstr: str = self._curtok.text
		self._get_next_token()  # eat identifier

		if self._curtok.type == EToken.ASSIGNMENT_OP:
			self._get_next_token()  # eat =

			if self._curtok.type | EToken.VALUE:
				ass_expr: AssignmentExpr = AssignmentExpr(idstr, self._curtok.value)
				self._get_next_token()  # eat value

				if self._curtok.type != EToken.STATEMENT_END:
					raise UnexpectedTokenException("%s: expected \";\"" % self._lexer.line_str)

				self._get_next_token()  # eat ;
				return ass_expr
			else:
				raise UnexpectedTokenException(
					"%s: \"%s\" is not a valid string, number, or boolean" % (self._lexer.line_str, self._curtok.text))

		elif self._curtok.type == EToken.BLOCK_START:
			if self._state == EParserState.BLOCK:
				raise UnexpectedTokenException("%s: nested blocks are not allowed" % self._lexer.line_str)
			else:
				self._state = EParserState.BLOCK

			self._get_next_token()  # eat {
			return BlockExpr(idstr, self._parse_block_expr())

		else:
			raise UnexpectedTokenException(
				"%s: expected \"=\" or \"{\" but got %s" % (self._lexer.line_str, self._curtok.text))

		pass

	def _parse_block_expr(self) -> List[AssignmentExpr]:
		ass_exprs: List[AssignmentExpr] = []

		while True:
			if self._curtok.type != EToken.BLOCK_END:
				ass_exprs.append(self._parse_identifier_expr())
			else:
				self._state = EParserState.NORMAL
				self._get_next_token()  # eat }
				break

		return ass_exprs


proper_names: Dict[str, str] = {
	"vertex_list": "vertices",
	"linedef_list": "linedefs",
	"sidedef_list": "sidedefs",
	"sector_list": "sectors",
	"thing_list": "things"
}


def parse_textmap(textmap: str, proper_key_names: bool = True) -> dict:
	"""
		Parses the UDMF textmap and returns its dictionary representation.
		textmap: the UDMF string to parse
		proper_key_names:	normally, this function appends "_list" to all objects constructed from block statements in
							UDMF. For instance, "vertex {x = 0.0, y = 0.0} vertex {x = 1.0, y = 1.0}" would translate
							to "vertex_list: [{x = 0.0, y = 0.0}, {x = 1.0, y = 1.0}]". This is because all
							unrecognized identifiers by the engine are ignored and one can't guarantee what the plural
							form of <identifier> is. This is true by default, meaning all recognized identifiers will
							be converted to its proper plural form. This includes "vertex_list", "linedef_list",
							"sidedef_list", "sector_list", "thing_list" to "vertices", "linedefs", "sidedefs",
							"sectors" and "things" respectively.
	"""
	result: dict = {}

	parser: Parser = Parser(textmap)
	for expr in parser:
		if isinstance(expr, AssignmentExpr):
			result[expr.key] = expr.value

		elif isinstance(expr, BlockExpr):
			key_str: str = expr.identifier + "_list"

			if proper_key_names:
				if key_str in proper_names:
					key_str = proper_names[key_str]

			if key_str not in result:
				result[key_str] = []

			result[key_str].append(expr.as_dictionary())

	return result


def convert_elements(elements, name):
	section = ''
	for element in elements:
		section += '\n' + name + '\n{\n'
		for key, val in element.items():
			val = f'"{val}"' if isinstance(val, str) else val  # Surround string values with quotes
			section += f'{key} = {val};\n'
		section += '}\n'
	return section


def convert_to_udmf(doom_map):
	namespace = doom_map['namespace']
	result = f'namespace = "{namespace}";\n'
	result += convert_elements(doom_map['things'], 'thing')
	result += convert_elements(doom_map['vertices'], 'vertex')
	result += convert_elements(doom_map['linedefs'], 'linedef')
	result += convert_elements(doom_map['sidedefs'], 'sidedef')
	result += convert_elements(doom_map['sectors'], 'sector')
	return result
