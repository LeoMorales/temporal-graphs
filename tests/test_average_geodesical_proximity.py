import unittest
import sys
sys.path.append('..')
import temporal_graph

class TestAverageGeodesicalProximity(unittest.TestCase):
    '''
        agp -> average geodesical proximity
    '''

    def setUp(self):
        self.graph = temporal_graph.get_temporal_graph_kostakos()

    def test_agp_A_D__is_2(self):
        self.assertEqual(
            self.graph.average_geodesic_proximity('A', 'D'),
            2
            )

    def test_agp_A_B__is_3_5(self):
        self.assertEqual(
            self.graph.average_geodesic_proximity('A', 'B'),
            3.5
            )

    def test_agp_E_B__is_4_5(self):
        self.assertEqual(
            self.graph.average_geodesic_proximity('E', 'B'),
            4.5
            )

    def test_agp_B_E__is_null(self):
        self.assertEqual(
            self.graph.average_geodesic_proximity('B', 'E'),
            None
            )


if __name__ == '__main__':
    unittest.main()