import sys
import pytest

default_args = ["-x", "--hypothesis-show-statistics"]
command_line_args = sys.argv[1:]


pytest.main(["--pyargs", "mappel",] + default_args + command_line_args)
