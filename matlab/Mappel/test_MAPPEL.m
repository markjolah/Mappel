import matlab.unittest.TestRunner;
import matlab.unittest.TestSuite;

suite = TestSuite.fromClass(?MappelTest);
runner = TestRunner.withTextOutput;

run(runner,suite)
