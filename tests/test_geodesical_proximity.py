import unittest
import sys
sys.path.append('..')
import temporal_graph

class TestGeodesicalProximity(unittest.TestCase):
    '''
        gp -> geodesical proximity
    '''

    def setUp(self):
        self.graph = temporal_graph.get_temporal_graph_kostakos()

    def test_gp_A_D_t1_null__is_3(self):
        self.assertEqual(
            self.graph.geodesic_proximity('A', 'D', 1, None),
            3
            )

    def test_gp_A_D_null_t5__is_4(self):
        ''' En el paper esta distancia figura como 5 (i.e. At1 At2 Et2 Et3 Dt3 Dt5).
        Pero hay un camino mas corto: 4 (i.e. At1 Bt1 Bt4 Bt5 Dt5)
        '''
        self.assertEqual(
            self.graph.geodesic_proximity('A', 'D', None, 5),
            4
            )

    def test_gp_A_D_t2_t5__is_2(self):
        self.assertEqual(
            self.graph.geodesic_proximity('A', 'D', 2, 7),
            2
            )

    def test_gp_A_D_null_null__is_1(self):
        self.assertEqual(
            self.graph.geodesic_proximity('A', 'D', None, None),
            1
            )


if __name__ == '__main__':
    unittest.main()