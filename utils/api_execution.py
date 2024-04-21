import googlesearch
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen

HEADERS ={'User-Agent': 'Mozilla/5.0'}

class ApiExecutor:
    def init(self, tool_name: str, tool_input: ):
        self.tool_name = tool_name
        self.tool_input = tool_input
    
    def execute(self, rtool_name, tool_input):
    