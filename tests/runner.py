# tests/runner.py
import unittest

# import your test modules
#import basic_example
import test_the_basics
import test_temporal_proximity
import test_average_temporal_proximity
import test_geodesical_proximity
import test_average_geodesical_proximity

def main():
	# initialize the test suite
	loader = unittest.TestLoader()
	suite  = unittest.TestSuite()

	# add tests to the test suite
	suite.addTests(loader.loadTestsFromModule(test_the_basics))
	suite.addTests(loader.loadTestsFromModule(test_temporal_proximity))
	suite.addTests(loader.loadTestsFromModule(test_average_temporal_proximity))
	suite.addTests(loader.loadTestsFromModule(test_geodesical_proximity))
	suite.addTests(loader.loadTestsFromModule(test_average_geodesical_proximity))
	
	# initialize a runner, pass it your suite and run it
	runner = unittest.TextTestRunner(verbosity=4)
	result = runner.run(suite)

if __name__ == '__main__':
	main()