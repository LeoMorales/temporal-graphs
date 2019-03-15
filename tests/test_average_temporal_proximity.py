import unittest
import sys
sys.path.append('..')
import temporal_graph

class TestAverageTemporalProximity(unittest.TestCase):
    '''
        atp -> average temporal proximity
    '''
    def setUp(self):
        self.graph = temporal_graph.get_temporal_graph_kostakos()

    def test_atp_A_D(self):
        self.assertEqual(
            temporal_graph.seconds_to_days(
                self.graph.average_temporal_proximity('A', 'D')),
            1.6666666666666667
            )

    def test_atp_A_A(self):
        self.assertEqual(
            temporal_graph.seconds_to_days(
                self.graph.average_temporal_proximity('A', 'A')),
            0
            )

    def test_atp_B_B(self):
        self.assertEqual(
            temporal_graph.seconds_to_days(
                self.graph.average_temporal_proximity('B', 'B')),
            0
            )

    def test_atp_A_B(self):
        self.assertEqual(
            temporal_graph.seconds_to_days(
                self.graph.average_temporal_proximity('A', 'B')),
            6.5
            )

    def test_atp_A_E(self):
        self.assertEqual(
            temporal_graph.seconds_to_days(
                self.graph.average_temporal_proximity('A', 'E')),
            0.5
            )

    def test_atp_B_E(self):
        self.assertEqual(
            temporal_graph.seconds_to_days(
                self.graph.average_temporal_proximity('E', 'B')),
            12
            )

    def test_atp_E_B(self):
        self.assertEqual(
            temporal_graph.seconds_to_days(
                self.graph.average_temporal_proximity('B', 'E')),
            None
            )


if __name__ == '__main__':
    unittest.main()