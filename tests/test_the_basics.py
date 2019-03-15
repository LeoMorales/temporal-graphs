import unittest
import sys
sys.path.append('..')
import temporal_graph

class TestBasics(unittest.TestCase):

    def setUp(self):
        self.graph = temporal_graph.get_temporal_graph_kostakos()

    def test_count_is_15_nodes(self):
        # the graph has 15 edges:
        self.assertEqual(self.graph.count(), 15)

    def test_size_is_18_edges(self):
        # the graph has 18 edges:
        self.assertEqual(self.graph.size(), 18)

if __name__ == '__main__':
    unittest.main()