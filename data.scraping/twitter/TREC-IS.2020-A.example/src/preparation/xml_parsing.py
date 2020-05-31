import xml.etree.cElementTree as ET
import pandas as pd


class XMLTopicsParser():

    def __init__(self, topics_path):
        self.topics_path = topics_path

    def parse_attribs(self, attribs) -> pd.DataFrame:
        xtree = ET.parse(self.topics_path)
        xroot = xtree.getroot()

        rows = []  # Parsed results are stored here

        for node in xroot: 
            res = []
            for el in attribs: 
                if node is not None and node.find(el) is not None:
                    res.append(node.find(el).text)
                    # Sanity check
                    # if el == 'num':
                    #     print(node.find('num').text)
                else: 
                    res.append(None)

            rows.append(
                {attribs[i]: res[i] for i, _ in enumerate(attribs)}
            )

        return pd.DataFrame(rows, columns=attribs)