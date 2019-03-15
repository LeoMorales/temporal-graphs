import unittest
import sys
sys.path.append('..')
import temporal_graph

class TestTemporalProximity(unittest.TestCase):
    '''
        tp -> temporal proximity
    '''

    def setUp(self):
        self.graph = temporal_graph.get_temporal_graph_kostakos()

    def test_tp_A_D_t2_t7(self):
        self.assertEqual(
            self.graph.temporal_proximity('A', 'D', 2, 7),
            ['A2', 'A7', 'D7']
            )

    def test_tp_A_D_t2_t7_weight_is_1641600_seconds(self):
        self.assertEqual(
            self.graph.weight(
                self.graph.temporal_proximity('A', 'D', 2, 7)),
            1641600.0
            )

    def test_tp_A_D_t2_t7_weight_is_19_days(self):
        self.assertEqual(
            temporal_graph.seconds_to_days(
                self.graph.weight(
                    self.graph.temporal_proximity(
                        'A', 'D', 2, 7))),
            19.0
            )

    def test_tp_A_D_t1_null__weight_equal_3(self):
        self.assertEqual(
            temporal_graph.seconds_to_days(
                self.graph.weight(
                    self.graph.temporal_proximity('A', 'D', 1, None))),
            3
            )

    def test_tp_A_D_null_t5__weight_equal_9(self):
        self.assertEqual(
            temporal_graph.seconds_to_days(
                self.graph.weight(
                    self.graph.temporal_proximity('A', 'D', None, 5))),
            9
            )

    def test_tp_A_D_null_null__weight_equal_0(self):
        self.assertEqual(
            temporal_graph.seconds_to_days(
                self.graph.weight(
                    self.graph.temporal_proximity('A', 'D', None, None))),
            0
            )

if __name__ == '__main__':
    unittest.main()